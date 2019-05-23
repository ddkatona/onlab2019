# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Implementation of the disentanglement metric from the BetaVAE paper.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework" (https://openreview.net/forum?id=Sy2fzU9gl).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
import os
from six.moves import range
from sklearn import linear_model
from disentanglement_lib.data.ground_truth.dsprites import DSprites
import tensorflow_hub as hub
import gin.tf
import tensorflow as tf
import random

@gin.configurable(
    "hungarian",
    blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_beta_vae_sklearn(ground_truth_data,
                             representation_function,
                             random_state,
                             #batch_size=gin.REQUIRED,
                             #num_train=gin.REQUIRED,
                             #num_eval=gin.REQUIRED,
                             model_dir=gin.REQUIRED):

  def random_data(amount):
    data = []
    for x in range(amount):
      x_index = random.randint(0,ground_truth_data.images.shape[0])
      data.append(ground_truth_data.images[x_index][..., np.newaxis])
    return np.array(data)

  def random_generated(amount):
    lat = np.random.normal(size=(amount,10))
    #print(lat)
    # Path to TFHub module of previously trained model.
    module_path = os.path.join(model_dir+"/model", "tfhub")
    with hub.eval_function_for_module(module_path) as f:
      output2 = f(dict(latent_vectors=lat), signature="decoder", as_dict=True)["images"]
      imgs = 1/(1+np.exp(-output2))
      return imgs

  def save_as_image(image, file_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.savefig(file_name)

  def string_to_image_pair(line, sd, gd):
    splitted_line = line.split()
    train_index = int(float(splitted_line[0]))
    test_index = int(float(splitted_line[1]))
    dst = splitted_line[2]
    return gd[train_index, :, :, 0], sd[test_index, :, :, 0], dst

  def read_pairs(lineList):
    records = []
    for i in range(0,len(lineList)):
      train_index = int(lineList[i].split()[0])
      test_index = int(lineList[i].split()[1])
      distance = float(lineList[i].split()[2])
      record = np.array([train_index, test_index, distance])
      records.append(record)
    return np.array(records)

  def find_image_by_index(index, images):
    return images[int(index), :, :, 0]

  def draw_random_pairs_from(pairs_file_path, sd, gd):
    fileHandle = open(pairs_file_path)
    lineList = fileHandle.readlines()
    n = len(lineList) # Number of pairs

    line = lineList[6]
    tr,te,dst = string_to_image_pair(line, sd, gd)
    save_as_image(tr, "train_image.png")
    save_as_image(te, "test_image.png")

    pairs = read_pairs(lineList)
    #print(pairs)
    sorted_pairs = pairs[pairs[:, 2].argsort()]
    #print(sorted_pairs)

    trains = []
    tests = []
    distances = []
    for i in range(0,len(sorted_pairs), len(sorted_pairs)//10):
    #for i in range(0,5):
      #rnd = random.randint(0,n)
      rnd = i
      
      train = find_image_by_index(sorted_pairs[rnd][0], gd)
      test = find_image_by_index(sorted_pairs[rnd][1], sd)
      trains.append(train)
      tests.append(test)
      distances.append(sorted_pairs[rnd][2])

    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(25, 8))
    columns = 2
    rows = len(trains)
    for i in range(1, columns*rows+1):
      if i <= len(trains):
        img = trains[i-1]
        distance = str(round(distances[i-1],2))
      else:
        img = tests[i-len(trains)-1]
        distance = ""
      fig.add_subplot(columns, rows, i)
      plt.imshow(img)
      plt.title(distance)
      plt.axis('off')
    plt.savefig("mx_test.png")
    fileHandle.close()

  data_count = 2000
  sampled_data = random_data(data_count)
  generated_data = random_generated(data_count)
  #save_as_image(sampled_data[0, :, :, 0], 'rdata.png')

  # Generate input for Hungarian method
  for x in range(data_count):
    np.savetxt("disentanglement_lib/evaluation/metrics/hungarian_lib/train/image_"+str(x)+".txt", generated_data[x, :, :, 0].flatten(),'%1.6f')
    np.savetxt("disentanglement_lib/evaluation/metrics/hungarian_lib/test/image_"+str(x)+".txt", sampled_data[x, :, :, 0].flatten(),'%1.6f')

  # Call Hungarian method
  root_path = 'disentanglement_lib/evaluation/metrics/hungarian_lib/'
  myCmd = root_path + "/bin/main -size 64,64 -folder1 " + root_path + "train -folder2 " + root_path + "test -N "+str(data_count)+" -range "+str(data_count)+" > " +root_path+ "out.txt"
  os.system(myCmd)
  fileHandle = open(root_path + "out.txt","r" )
  lineList = fileHandle.readlines()
  fileHandle.close()
  min_distance = float(lineList[-1].split()[-1])/data_count # result

  # Handle random pairs
  draw_random_pairs_from(root_path + "mnist_result_"+str(data_count)+".txt", sampled_data, generated_data)
  """fileHandle = open(root_path + "mnist_result_"+str(data_count)+".txt")
  lineList = fileHandle.readlines()
  print(lineList[5])
  fileHandle.close()"""

  logging.info("Evaluate evaluation set accuracy.")
  #eval_accuracy = 0.99
  logging.info("Minimum distance: " + str(min_distance))
  scores_dict = {}
  #scores_dict["train_accuracy"] = train_accuracy
  scores_dict["min_distance"] = min_distance
  return scores_dict


def _generate_training_batch(ground_truth_data, representation_function,
                             batch_size, num_points, random_state):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.

  Returns:
    points: (num_points, dim_representation)-sized numpy array with training set
      features.
    labels: (num_points)-sized numpy array with training set labels.
  """
  points = None  # Dimensionality depends on the representation function.
  labels = np.zeros(num_points, dtype=np.int64)
  for i in range(num_points):
    labels[i], feature_vector = _generate_training_sample(
        ground_truth_data, representation_function, batch_size, random_state)
    if points is None:
      points = np.zeros((num_points, feature_vector.shape[0]))
    points[i, :] = feature_vector
  return points, labels


def _generate_training_sample(ground_truth_data, representation_function,
                              batch_size, random_state):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample
    random_state: Numpy random state used for randomness.

  Returns:
    index: Index of coordinate to be used.
    feature_vector: Feature vector of training sample.
  """
  # Select random coordinate to keep fixed.
  index = random_state.randint(ground_truth_data.num_factors)
  # Sample two mini batches of latent variables.
  factors1 = ground_truth_data.sample_factors(batch_size, random_state)
  factors2 = ground_truth_data.sample_factors(batch_size, random_state)
  # Ensure sampled coordinate is the same across pairs of samples.
  factors2[:, index] = factors1[:, index]
  # Transform latent variables to observation space.
  observation1 = ground_truth_data.sample_observations_from_factors(
      factors1, random_state)
  observation2 = ground_truth_data.sample_observations_from_factors(
      factors2, random_state)
  # Compute representations based on the observations.
  representation1 = representation_function(observation1)
  representation2 = representation_function(observation2)
  # Compute the feature vector based on differences in representation.
  feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
  return index, feature_vector
