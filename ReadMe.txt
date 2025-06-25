- For novel model training, some of the folds in the training go to NaN, this could be for multiple reasons, including the semi supervised nature of the training, so the calculation of the
uncertainty head calculating the log variance, the variance could be approximately equal to zero if the masks are very biased to mostly 1s or mostly 0s making log variance 
tend to infinity. Easy solution would be to clamp these values and the generator adversarial and discriminator adversarial losses as well, the code is sound but certain batches 
with corrupted data could be the cause of this. So if the rain image is a mask that is already grayscale so the output of the mask network creates a very light or dark image,
so this training technique could reveal some corruption in the data.
-Pipeline is fine and due to time constraints I will take the checkpoint weights for folds where losses to nan right before they turned and compare the validation metrics of those
but a quick fix is to clamp losses that might be the cause of this between local minimum and maximum to avoid the blow up. 