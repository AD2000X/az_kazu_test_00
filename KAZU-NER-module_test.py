#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging                           # import the log module, used to record information when the program is running
import os                                # import modules for operating system interfaces, such as reading and writing files
import sys                               # import modules for accessing variables and functions closely related to the Python interpreter
from dataclasses import dataclass, field # import the module used to create data classes
from typing import Optional              # import module for annotating optional types

import datasets                          # import the datasets library, usually used to load and process datasets
import numpy as np                       # import the NumPy library, which provides functions that support
                                         #  large-dimensional array and matrix operations
from datasets import ClassLabel, load_dataset, load_metric  # import specific functions and classes from the datasets library

import transformers                      # import Hugging Face’s transformers library
from transformers import (               # import specific classes and functions from the transformers library
    AutoConfig,                          # automatically load model configuration
    AutoModelForTokenClassification,     # automatically load the tokenizer suitable for the model
    AutoTokenizer,                       # automatically load the tokenizer suitable for the model
    DataCollatorForTokenClassification,
    HfArgumentParser,                    # parser for parsing command line arguments
    PreTrainedTokenizerFast,             # fast tokenizer, suitable for pre-trained models
    Trainer,                             # trainer class, used to train models
    TrainingArguments,                   # training parameter class, used to set training configuration
    set_seed,                            # set a random seed so that the experiment can be reproduced
)
from transformers.trainer_utils import get_last_checkpoint  # function used to get the last checkpoint
from transformers.utils import check_min_version            # function for checking transformers library version
from transformers.utils.versions import require_version     # force a specific version of a function

# from multi_label_trainer import MultiLabelProbTrainer
# for multi-label token classification tasks. In multi-label classification, each token (e.g., word or character in text)
#  can belong to multiple categories simultaneously.
# A data aggregator for processing data from probabilistic token classification tasks. It may preprocess the input data,
#  such as encoding, batching, adding necessary padding or masking, so that it can be processed correctly by the model.
from modeling import BertForMultiLableTokenClassification
from data_collator import DataCollatorForProbTokenClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.12.0", "To fix: pip install -r requirements.txt")  # HF changed the API after v1.11

# initializing a logger object from the logging module using the name of the current module.
# logging: This is the standard Python module for implementing flexible event logging for applications and libraries
# getLogger(): This method is called from the logging module to create or retrieve a logger instance.
#  If you provide a name, you get a logger with that name; if no name is provided, you get the root logger
# __name__: This is a special Python variable that holds the name of the current module. If this line is within a script
#  that is being run directly, then __name__ will be '__main__', but if it's in an imported module, __name__ will be
#  the name of that module
# Write messages to the configured log destination (console, file, etc.), with varying levels of
#  severity (DEBUG, INFO, WARNING, ERROR, and CRITICAL).
logger = logging.getLogger(__name__)

# a decorator that automatically generates special methods for the class, such as __init__(), __repr__(),
#  and others, based on the class attributes
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # model_name_or_path: This is a class attribute of type str that holds either the path to a pretrained model or a model
    #  identifier from the Hugging Face model repository.
    # field(): This function is used to provide additional information about the class attribute. Here, it's used to
    #  provide a help string in the metadata, which can be used by a command-line interface to explain the argument.
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # config_name: Another attribute of type Optional[str], which means it can be either a string or None.
    #  It's used to specify a path to a model configuration file if it's different from the model name.
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # tokenizer_name: Similar to config_name, this attribute specifies the tokenizer to use and has the same type expectations.
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # cache_dir: An attribute that specifies where to store downloaded pretrained models from Hugging Face.
    #  This is useful for managing disk space and avoiding repeated downloads
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # model_revision: An attribute that allows specifying a version of the model, which can be useful for
    #  reproducibility or when working with models that have multiple versions
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # use_auth_token: A boolean attribute that, when set to True, indicates that the script should use an authentication
    #  token. This is required when accessing private models or APIs that need authentication.
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # task_name: An optional string that specifies the task you're training the model for
    #  (e.g., named entity recognition (NER), part-of-speech tagging (POS), etc.). It defaults to "ner".
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    # dataset_name: An optional string that specifies the name of the dataset to use, which should be available in the
    #  Hugging Face datasets library.
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    # dataset_config_name: An optional string to specify the configuration name of the dataset, in case the dataset
    #  has multiple configurations (like different languages or versions).
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    # train_file: An optional string that specifies the path to the training data file, which can be in CSV or JSON format.
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    # validation_file: Similar to train_file, this specifies the path to the validation data file.
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    # test_file: This specifies the path to the test data file if there is one.
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    # text_column_name: An optional string that specifies the column in the data file that contains the text data for training.
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    # label_column_name: This attribute is required and specifies the column name in the data file that contains
    #  the labels for the text data.
    label_column_name: str = field(
        default=None, metadata={
            "help": "The column name of label to input in the file REQUIRED! Should be one of labels or label_probs."}
    )
    # overwrite_cache: A boolean flag indicating whether to overwrite the cached data sets.
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    # preprocessing_num_workers: An optional integer specifying the number of worker processes to use during data preprocessing.
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # pad_to_max_length: A boolean flag that indicates whether to pad all the samples to the maximum length supported by the model.
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    # max_train_samples: An optional integer to limit the number of training samples, useful for quicker training or debugging.
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    # max_eval_samples: This is an optional integer that allows you to limit the number of evaluation samples.
    #  This can be useful for debugging or for times when you want to speed up evaluation.
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    # max_predict_samples: Similar to max_eval_samples, but for prediction. If set, it limits the number of
    #  prediction samples, which can speed up the prediction phase, especially during development.
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    # label_all_tokens: A boolean that determines if every sub-token of a split word gets the same label in token
    #  classification tasks. If False, only the first token of a split word is labeled, and the rest are labeled with a padding index.
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
                    "one (in which case the other tokens will have a padding index)."
        },
    )
    # return_entity_level_metrics: A boolean indicating whether to return detailed metrics for each entity type during
    #  evaluation or just overall metrics.
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    # max_length: An integer that sets the maximum number of tokens for the input sequences. If sequences are longer
    #  than this number, they will be truncated to this length.
    max_length: int = field(
        default=128,
        metadata={
            "help": "The specific the maximum length of input sequence (the number of tokens) that model handles."
                    "See https://huggingface.co/docs/transformers/pad_truncation for details."
                    "Current setting: truncation to specific length and no padding."
        },
    )
    # use_probs: A boolean indicating if the token classification task should be treated as a multi-label problem,
    #  where each token can belong to multiple classes.
    use_probs: bool = field(
        default=False,
        metadata={
            "help": "This will activate multi-label token classification."
                    "Affected: tokenization steps, modeling"
        },
    )
    # local_files_only: A boolean that, if True, will restrict the use of files to those already present on your local
    #  machine without attempting to download them from the internet.
    local_files_only: bool = field(
        default=False,
        metadata={
            "help": "This is for cache"
        },
    )
    # __post_init__, is called automatically after the DataTrainingArguments class is initialized.
    #  It is used to validate the arguments passed to the class.
    def __post_init__(self):
        # check if neither dataset_name nor file paths for training and validation data (train_file and validation_file)
        #  are provided.
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            # If all of them are None, it raises a ValueError indicating that at least one of these must be specified
            #  for the training or validation process to proceed.
            raise ValueError("Need either a dataset name or a training/validation file.")
        # else clause is executed if the above condition is not met, which means there is at least a dataset name or
        #  file path provided.
        else:
            # check if a train_file has been provided.
            if self.train_file is not None:
                # It retrieves the file extension of the training file by splitting the string on the period and
                #  taking the last element.
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "prob_conll"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                # gets the file extension of the validation file in the same way as it did for the training file.
                extension = self.validation_file.split(".")[-1]
                # assertion makes sure that the validation_file has a valid file extension.
                assert extension in ["csv", "json", "prob_conll"], "`validation_file` should be a csv or a json file."
        # converts the task_name to lowercase to avoid case sensitivity issues.
        self.task_name = self.task_name.lower()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    # Initializes an HfArgumentParser object from the transformers library, which is designed to handle command-line
    #  arguments. The parser is configured to expect arguments defined in the ModelArguments, DataTrainingArguments,
    #  and TrainingArguments data classes.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if statement checks if exactly one command-line argument is provided (in addition to the script name) and if this
    #  argument (a path) ends with ".json". This is used to determine if the user intends to pass arguments via a JSON file.
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # If the condition is true, it parses the JSON file specified in the command line into model_args, data_args,
        #  and training_args using the parse_json_file method of the parser. os.path.abspath(sys.argv[1]) gets the
        #  absolute path of the JSON file.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # If the JSON file condition is not met, it falls back to parsing the command-line arguments into the three sets
    #  of arguments using the parse_args_into_dataclasses method of the parser.
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    # This sets up basic configuration for the logging system. The log messages will include the timestamp,
    #  severity level, logger's name, and the message itself. The datefmt specifies the format of the timestamp.
    #  Messages are directed to standard output (sys.stdout) using a stream handler.
    # python
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # The log level (e.g., INFO, WARNING, ERROR) is obtained from the training_args and set for the logger.
    #  This determines the minimum severity of messages the logger will handle.
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # These lines set the logging verbosity for the datasets and transformers libraries respectively to the same
    #  level as determined for the main logger.
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    # Enables the default logging handler and explicit formatting for the transformers library. This is to ensure
    #  consistent logging behavior and format across different parts of the application.
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # Logs a warning with a summary of the training setup, including process rank, device, number of GPUs,
    #  whether distributed training is enabled, and if 16-bits (FP16) training is used.
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Logs the training and evaluation parameters at the INFO level for reference.
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    # Checks if the output directory exists, if training is enabled (do_train=True), and if the output directory is
    #  not set to be overwritten. This is to detect if there's a last checkpoint to resume training from.
    # If conditions are met, attempts to get the last checkpoint from the output directory.
    last_checkpoint = None
    # Checks if no last checkpoint is found but the output directory is not empty.
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # Raises a ValueError indicating the output directory already exists and is not empty.
            #  It suggests using --overwrite_output_dir to allow overwriting.
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        # If a last checkpoint is found and there's no instruction to resume from a specific checkpoint,
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            # Logs information that a checkpoint has been detected and training will be resumed from it.
            #  It advises how to train from scratch instead if desired.
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if "labels" in column_names:
        assert isinstance(features["labels"].feature, ClassLabel)
        label_list = features["labels"].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    label_to_index = {l: i for i, l in enumerate(label_list)}

    data_args.num_labels = num_labels
    # model_args.num_labels = num_labels

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            truncation=True,
            max_length=data_args.max_length,  # TODO: For some reasons, this line dose not work
            local_files_only=data_args.local_files_only,  # to avoid tokenizing again
        )

    model = BertForMultiLableTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            max_length=data_args.max_length
        )
        labels = []
        label_probs = []  # should contain # of samples in the preproc-batch

        if data_args.use_probs:
            iter_examples = zip(examples[label_column_name], examples["label_probs"])
        else:
            iter_examples = examples[label_column_name]

        for i, single_sample in enumerate(iter_examples):
            if data_args.use_probs:
                label, label_prob_inp = single_sample
            else:
                label = single_sample
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            label_prob_outputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)

                if data_args.use_probs:
                    if word_idx is None:
                        label_prob_outputs.append([0] * data_args.num_labels)
                    elif word_idx != previous_word_idx:
                        label_prob_outputs.append(label_prob_inp[word_idx])
                    else:
                        label_prob_outputs.append(
                            label_prob_inp[word_idx] if data_args.label_all_tokens else [0] * data_args.num_labels)

                previous_word_idx = word_idx

            labels.append(label_ids)
            label_probs.append(label_prob_outputs)
        tokenized_inputs["labels"] = labels
        if data_args.use_probs:
            tokenized_inputs["label_probs"] = label_probs
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # for sanity checking
    logger.info(f"#### Printing pre-processed samples, each from train_dataset and predict_dataset.")
    show_sample = [train_dataset] if "train" in raw_datasets else []
    if "test" in raw_datasets:
        show_sample.append(predict_dataset)

    for sample_data_type in show_sample:
        for sample_idx in [0, 1]:
            for key, values in sample_data_type[sample_idx].items():
                logger.info(f"{key} ({len(values)}): {values}")
    logger.debug("labels\ttoken_type_ids input_ids")
    for values in zip(
            sample_data_type[0]["labels"],
            sample_data_type[0]["token_type_ids"],
            sample_data_type[0]["input_ids"],
    ):
        logger.debug("\t".join([str(ele) for ele in values]) + " : " + tokenizer.decode(values[2]))

    if not (data_args.use_probs):
        logger.critical("######################\n" * 2 + \
                        "## data_args.use_probs not selected! Check wheater this is what you intended! ##\n" + \
                        "######################\n" * 2)

    # Data collator
    # data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    data_collator = DataCollatorForProbTokenClassification(tokenizer,
                                                           pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    # metric = load_metric("seqeval", cache_dir=training_args.output_dir)
    metric = load_metric("evaluate_multi_label.py", cache_dir=training_args.output_dir)

    def compute_metrics(p):
        # predictions_prob, labels, label_probs = p
        predictions_prob, labels = p
        predictions = np.argmax(predictions_prob, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # results = metric.compute(predictions=true_predictions, references=true_labels, label_list=label_list) # for label level

        true_predictions_prob = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions_prob, labels)
        ]
        # pseudo-true-label generated using true labels (TODO: use label_probs)
        true_label_probs = [
            np.eye(len(label_list))[[label_to_index[l] for l in label]]
            for label in true_labels
        ]
        results = metric.compute(predictions=true_predictions_prob, references=true_label_probs, label_list=label_list)

        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Initialize our Trainer
    trainer = Trainer(  # MultiLabelProbTrainer
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()