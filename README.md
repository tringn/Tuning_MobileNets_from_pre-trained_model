# Introduction
This section wil describe how to fine-tune your own model from the existing checkpoint of Mobilenet V2. This tutorial is referenced 
from https://github.com/tensorflow/models/tree/master/research/slim#fine-tuning-a-model-from-an-existing-checkpoint with some changes
to train your own model.
The example training dataset is sat dataset, which is used to recognize worker and customer.

# Prerequisites
Package Requirements:
```ruby
- tensorflow>=1.7.0,<=1.8.0
```
# Installation
# I/ Prepare dataset to train
Create training directory ./data/sat/sat_photos. Then download sat dataset photos to train. Each category is stored in a separated folder whose name is "class" to describe. (e.g. customer, worker).
Save these folders in ./data/sat/sat_photos/
Dataset storage will be in that structure:
```
data
└── sat
    └── sat_photos
        ├── customer
        └── worker
```

# II/ Modify set-up file to convert dataset to TRRecord format
Since this set-up is prepared by Tensorflow to train / tuning 4 dataset (Flower, Cifar10, MNIST, ImageNet), we have to modify some set-up files to tuning our own model.
I've already trained my own model called poses. Now you can modify my set-up to train your own model.
1. Copy datasets/convert_poses.py to datasets/convert_sat.py
```ruby
cp datasets/convert_poses.py datasets/convert_sat.py
```
2. Open file datasets/convert_sat.py, edit **__NUM_VALIDATION_** variable to desired number of validation photos.
This number depends on the size of your dataset. Then replace all words "poses" by "sat" in this file.
3. Open file download_and_convert_data.py
- Add **_from datasets import convert_sat_**
- Add **_sat_** to **_tf.app.flags.DEFINE_string_**
- Add command **_convert_sat.run(FLAGS.dataset_dir)_** to "if" in **_main(_)**

# III/ Convert to TFRecord Format
For each dataset, we need to convert raw data to TensorFlow's native TFRecord format. Each TFRecord contains a TF-Example protocol buffer.
```ruby
python3 download_and_convert_data.py --dataset_name=sat --dataset_dir=./data/sat
```
# IV/ Modify set-up file to train model
1. Copy datasets/poses.py to datasets/sat.py.
```ruby
cp ./datasets/poses.py ./datasets/sat.py
```
2. Open file datasets/sat.py, 
- Edit **_SPLITS_TO_SIZES_** to the number of photos used for training and validation.
- Edit **__NUM_CLASSES_** to 2 (because there are 2 classes: customer and worker). 
- Edit **__FILE_PATTERN_** to **_sat_%s_*.tfrecord_**
- Replace all words "poses" by "sat"
3. Open file datasets/dataset_factory.py
- Add **_from datasets import sat_**
- Add **_'sat': sat,_** to datasets_map

# V/ Download Checkpoint of pre-trained model of Mobilenet V2 and fine-tune your own model
1. Download Checkpoint of MobilenetV2 pre-trained model
```ruby
mkdir ./my_checkpoints
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz 
mv mobilenet_v2_1.4_224.tgz my_checkpoint
tar -xvf mobilenet_v2_1.4_224.tgz
rm mobilenet_v2_1.4_224.tgz
```
2. Fine-tune your own model
To indicate a checkpoint from which to fine-tune, we'll call training with the --checkpoint_path flag and assign it an absolute path to a checkpoint file.

When fine-tuning a model, we need to be careful about restoring checkpoint weights. In particular, when we fine-tune a model on a new task with a different number of output labels, we won't be able restore the final logits (classifier) layer. For this, we'll use the --checkpoint_exclude_scopes flag. This flag hinders certain variables from being loaded. When fine-tuning on a classification task using a different number of classes than the trained model, the new model will have a final 'logits' layer whose dimensions differ from the pre-trained model. For example, if fine-tuning an ImageNet-trained model on sat dataset, the pre-trained logits layer will have dimensions [2048 x 1001] but our new logits layer will have dimensions [2048 x 2]. Consequently, this flag indicates to TF-Slim to avoid loading these weights from the checkpoint.

Keep in mind that warm-starting from a checkpoint affects the model's weights only during the initialization of the model. Once a model has started training, a new checkpoint will be created in --train_dir. If the fine-tuning training is stopped and restarted, this new checkpoint will be the one from which weights are restored and not the --checkpoint_path. Consequently, the flags --checkpoint_path and --checkpoint_exclude_scopes are only used during the 0-th global step (model initialization). Typically for fine-tuning one only want train a sub-set of layers, so the flag --trainable_scopes allows to specify which subsets of layers should trained, the rest would remain frozen.

Below we give an example of fine-tuning Mobilenet-v2 on sat, Mobilenet_v2 was trained on ImageNet with 1000 class labels, but the sat dataset only have 2 classes. Since the dataset is quite small we will only train the new layers.
```ruby
python3 train_image_classifier.py \
    --train_dir=./sat_models/mobilenet_v2 \
    --dataset_dir=./data/sat \
    --dataset_name=sat \
    --dataset_split_name=train \
    --model_name=mobilenet_v2 \
    --checkpoint_path=my_checkpoints/mobilenet_v2_1.4_224.ckpt \
    --checkpoint_exclude_scopes=MobilenetV2/Logits \
    --trainable_scopes=MobilenetV2/Logits \
    --max_number_of_steps=50000 \
    --batch_size=32 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004 \
    --train_image_size=224
```
For more information about Gradient Descent optimizer algorithm, you can refer at: http://ruder.io/optimizing-gradient-descent/index.html

# VI/ Evaluating performance of a model
To evaluate the performance of a model, you can use the eval_image_classifier.py script, as shown below.
```ruby
python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=./sat_models/mobilenet_v2/model.ckpt-50000 \
    --dataset_dir=./data/sat \
    --dataset_name=sat \
    --dataset_split_name=validation \
    --model_name=mobilenet_v2 \
    --eval_image_size=224
```
# VII/ Exporting the Inference Graph
Saves out a GraphDef containing the architecture of the model.
To use it with a model name defined by slim, run:
```ruby
python3 export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v2 \
  --output_file=./sat_mobilenet_v2_inf.pb \
  --dataset_name=sat \
  --image_size=224
```
# VIII/ Freezing the exported Graph
  If you then want to use the resulting model with your own or pretrained checkpoints as part of a mobile model, you can run freeze_graph to get a graph def with the variables inlined as constants using:
  ```ruby
python3 freeze_graph.py \
--input_graph=./sat_mobilenet_v2_inf.pb \
--input_checkpoint=./sat_models/mobilenet_v2/model.ckpt-50000 \
--input_binary=true \
--output_graph=./sat_mobilenet_v2_frz.pb \
--output_node_names=MobilenetV2/Predictions/Reshape_1
  ```
# IX/ Test your model
Run the code below:
```ruby
python3 label_image.py \
--graph=./sat_mobilenet_v2_frz.pb \
--labels=./data/sat/labels.txt \
--image=<PATH_TO_IMAGE>
```
 
# Reference
- Fine-tuning a model from an existing checkpoint: https://github.com/tensorflow/models/tree/master/research/slim#fine-tuning-a-model-from-an-existing-checkpoint
- Gradien Descent optimizer algorithm: http://ruder.io/optimizing-gradient-descent/index.html

