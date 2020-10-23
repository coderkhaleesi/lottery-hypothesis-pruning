# lottery-hypothesis-pruning

Abstract. Facial expression recognition (FER) models can be used to
automatically detect a person’s emotions based on an image of their
face. In this work we create a FER model by fine-tuning an ImageNetpretrained ResNet-18 on the Static Facial Expression in the Wild (SFEW)
dataset. We aim to investigate how well network pruning methods can
function on this fine-tuned FER model. We compare the performance of
two different pruning algorithms, magnitude pruning and distinctiveness
pruning, and find that magnitude pruning performs significantly better.
In-fact, we find that after pruning away 50% of the weights in the
network, the network can continue to be trained and reach an even higher
test accuracy (66.14%) compared to the original full model (63.36%). In
addition, in order to confirm that the lottery hypothesis still holds for
fine-tuned networks, we perform experiments where un-pruned weights
are re-initialized to their original pre-trained value after pruning. Our
experiments successfully demonstrate that there exist subnetworks in
the pre-trained ResNet with as little at 5% of the total weights in the
full network which can be successfully trained to the same performance
as the full model.
