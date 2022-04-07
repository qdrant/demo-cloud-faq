Quaterion library is actually split into two separate libraries: Quaterion and Quaterion models.
The responsibility of the latter is serving of trained models. It contains only the most
necessary modules, such as encoders, heads and MetricModel itself. You have no need to bring all 
training related stuff into production. Hence, it led to reduction of memory footprint and number
of entities you need to operate for serving. And Quaterion library is the one responsible for a 
training process.

Main entity in Quaterion library is TrainableModel.
TrainableModel is a wrapper around LightningModule. Lightning handles all the training process 
complexities and saves user from a necessity to implement it manually. Also, one of important 
Lightning's benefits is its modularity. Modularity improves separation of responsibilities, makes 
code more readable, robust and easy to write. And as you can guess, separation of responsibilities
is one of Quaterion's features. Nevertheless, it inherits other features too. 
All  the steps you need to follow to train a model are on the surface with TrainableModel. You will 
have to implement a few methods like configure_loss, configure_encoders and configure_head to make 
your minimal metric learning model! 

When you are done with your experiments - there is a short road to use your model for serving:
you just need to implement save methods in your encoders and call save_servable from your model. 
From that point everything you need is Quaterion-models, no need in Quaterion itself anymore.
To obtain embeddings one need to load trained model and call encode method with a batch of data,
that's it, your embeddings are ready for search.


During training process you need to feed your model with data. In metric learning there are usually 
no labels, only knowledge if some objects are similar or dissimilar. And for better data 
representation and standardization in this situation, Quaterion introduces SimilaritySample and its
heirs - SimilarityPairSample and SimilarityGroupSample.

In SimilarityPairSample consists of two objects which are similar, or dissimilar - obj_a and obj_b,
whether they are similar or not one points via score field. Currently, score usually being 
transformer from float number into a bool, but one can also attend some pairs be more important
than the others. However, there is no any ready losses or metrics to account it and if one wants
to leverage such a feature, some code should be written for it. The last, but not the least field
in SimilarityPairSample is subgroup. All pairs, which are not in the same subgroup as the current
pair, could be considered as negative samples.

For example, in question-answering task a positive pair (a pair of similar objects) might be a 
couple of a question and its answer. A couple of a question and an irrelevant answer or other 
question might be an example of a negative pair. 

Unlike SimilarityPairSample, SimilarityGroupSample consists only of one object field and group id.
All objects from the same group will be considered as similar, the others - dissimilar.

To configure loss for a model, you can choose a ready loss class from already implemented in 
Quaterion. However, since Quaterion's purpose is not to cover all possible losses, or tactics for 
mining negative samples, etc., but to provide a convenient framework to build and use metric 
learning models, there might not be a desired loss. In this case it is possible to use 
PytorchMetricLearningWrapper to bring required loss from pytorch-metric-learning library, which has
more rich collection of losses, or implement a particular loss yourself.

Quaterion also provides functionality to measure distances via different approaches. Cosine,
euclidean, manhattan distances and even an interpretation of dot_product as a distance are 
available out-of-the-box via Distance class.

One of the further steps is to configure encoders to be able to calculate embeddings. Encoder is
a model which accepts some input and emits an embedding. In most of the situations you will use
one pre-trained model as an encoder, but there is also a possibility to configure several encoders 
simultaneously or train them yourself.

Encoders prone to time-consuming computations, but in case if they are frozen - there is a tool
to mitigate their heaviness. If encoders are frozen, they are deterministic and emits exactly the
same embeddings for the same input data each epoch. So why not to avoid this and reduce training
time? For this purpose Quaterion has a cache. Before training starts, cache runs one epoch to 
calculate all embeddings from frozen encoders and then store them on a device you chose
(currently CPU or GPU). Everything you need to do is to define which encoders are trainable
and which are not and set cache settings. And that's it: everything else Quaterion will handle
for you.





