# Copyright 2019 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import preprocessor.tokenization as tokenization

max_seq_length = 128

class IntentSlotExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, intent_label=None, slot_labels=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.intent_label = intent_label
    self.slot_labels = slot_labels


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class IntentSlotFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               intent_label_id,
               slot_label_ids,
               slot_label_mask,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.intent_label_id = intent_label_id
    self.slot_label_ids = slot_label_ids
    self.slot_label_mask = slot_label_mask
    self.is_real_example = is_real_example


def convert_single_example(ex_index, example, intent_label_list, slot_label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  #max_se_length-2: spot length does not include [CLS] + [SEP]
  if isinstance(example, PaddingInputExample):
    return IntentSlotFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        intent_label_id=0,
        slot_label_ids=[0] * (max_seq_length-2),
        slot_label_mask=[0] * (max_seq_length-2),
        is_real_example=False)

  intent_label_map = {}
  for (i, intent_label) in enumerate(intent_label_list):
    intent_label_map[intent_label] = i

  slot_label_map = {}
  for (i, slot_label) in enumerate(slot_label_list):
    slot_label_map[slot_label] = i

  #Slot_label_mask indicates starting word piece token which correspond to slot labels
  tokens_a = []
  slot_label_mask = []
  slot_label_to_token_map = []
  for token in tokenizer.basic_tokenizer.tokenize(example.text_a):
    count = 0
    slot_label_to_token_map.append(len(tokens_a))
    for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
      tokens_a.append(sub_token)
      count = count + 1
    slot_label_mask.append(1)
    if count > 1:
      slot_label_mask.extend([0] * (count-1))

  tokens_b = None
  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]
    slot_label_mask = slot_label_mask[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  intent_label_id = intent_label_map[example.intent_label]
  #put labels according to map resulted from word piece tokenization
  slot_label_ids = [0] * (max_seq_length - 2)
  for i, slot_label in enumerate(example.slot_labels):
    slot_label_ids[slot_label_to_token_map[i]] = slot_label_map[slot_label]

  # Zero-pad slot label mask.
  while len(slot_label_mask) < max_seq_length - 2:
    slot_label_mask.append(0)

  assert len(slot_label_mask) == max_seq_length - 2

  #if ex_index < 5:
  #  tf.logging.info("*** Example ***")
  #  tf.logging.info("guid: %s" % (example.guid))
  #  tf.logging.info("text_a: %s" % (example.text_a))
  #  tf.logging.info("tokens: %s" % " ".join(
  #      [tokenization.printable_text(x) for x in tokens]))
  #  tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
  #  tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
  #  tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
  #  tf.logging.info("intent label: %s (id = %d)" % (example.intent_label, intent_label_id))
  #  tf.logging.info("slot label ids: %s" % " ".join([str(x) for x in slot_label_ids]))
  #  tf.logging.info("slot labels: %s" % " ".join([x for x in example.slot_labels]))
  #  tf.logging.info("slot label mask: %s" % " ".join([str(x) for x in slot_label_mask]))
  #  tf.logging.info("slot label token map: %s" % " ".join([str(x) for x in slot_label_to_token_map]))

  feature = IntentSlotFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      intent_label_id=intent_label_id,
      slot_label_ids=slot_label_ids,
      slot_label_mask=slot_label_mask,
      is_real_example=True)
  return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class BertPreprocessor(object):
    """Stores means of each column of a matrix and uses them for preprocessing.
    """

    def __init__(self, vocab_file, intent_labels, slot_labels, max_seq_length=128, do_lower_case=True):
        """On initialization, is not tied to any distribution."""
        self._max_seq_length=max_seq_length
        self._tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self._intent_label_list = []
        self._slot_label_list = []

        with open(intent_labels, 'r') as in_file:
          for l in in_file:
            self._intent_label_list.append(l.strip())

        with open(slot_labels, 'r') as in_file:
          for l in in_file:
            self._slot_label_list.append(l.strip())

    def get_labels(self):
      """See base class."""
      return (self._intent_label_list, self._slot_label_list)


    def get_words(self, data):
      """See base class."""
      words = []
      for item in data:
        words.append(self._tokenizer.basic_tokenizer.tokenize(item["text_a"]))

      return words


    def get_tokens(self, data):
      """See base class."""
      tokens = []
      for item in data:
        tokens.append(self._tokenizer.tokenize(item["text_a"]))

      return tokens


    def preprocess(self, data):
        """Transforms a matrix.

        The first time this is called, it stores the means of each column of
        the input. Then it transforms the input so each column has mean 0. For
        subsequent calls, it subtracts the stored means from each column. This
        lets you 'center' data at prediction time based on the distribution of
        the original training data.

        Args:
            data: A NumPy matrix of numerical data.

        Returns:
            A transformed matrix with the same dimensions as the input.
        """

        features = []
        for i, item in enumerate(data):
            example = IntentSlotExample(
                guid="%s" % (i),
                text_a=tokenization.convert_to_unicode(item["text_a"]),
                text_b=None,
                intent_label=self._intent_label_list[0],
                slot_labels=[self._slot_label_list[0]]
            )

            feature = convert_single_example(i, example, self._intent_label_list, self._slot_label_list, self._max_seq_length, self._tokenizer)

            features.append({
                "input_ids" : feature.input_ids,
                "input_mask" : feature.input_mask,
                "segment_ids" : feature.segment_ids,
                "intent_label_ids" : feature.intent_label_id,
                "slot_label_ids" : feature.slot_label_ids,
                "slot_label_mask" : feature.slot_label_mask
            })
        
        return features
