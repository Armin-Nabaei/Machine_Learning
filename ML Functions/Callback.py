from tensorflow.keras.optimizers import RMSprop, SGD, Adam,Adadelta,Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,LearningRateScheduler
from keras.models import load_model
import math

# Setup EarlyStopping callback to stop 
# training if model's val_loss
# doesn't improve for 3 epochs
 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=15) # if val loss decreases for 3 epochs in a row, stop training

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.96, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=3,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)
                                                 

def decay(epochs, steps=100):
    initial_lrate = 0.0001
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+(2*epochs)/epochs_drop)))
    return lrate


initial_lrate = 0.0001 

checkpoint = tf.keras.callbacks.ModelCheckpoint("/content/drive/MyDrive/ATTENTION/cifar10_1.h5",
                             #save the best model while all epovhs of training
                             monitor="val_accuracy",
                             save_best_only=True, # only save the best weights
                             save_weights_only=True, # only save model weights (not whole model)                            
                             
                             verbose=1)


#https://stackoverflow.com/questions/47626254/changing-optimizer-in-keras-during-training

class OptimizerChanger(EarlyStopping):

    def __init__(self, on_train_end, **kwargs):

        self.do_on_train_end = on_train_end
        super(OptimizerChanger,self).__init__(**kwargs)

    def on_train_end(self,logs=None):
        super(OptimizerChanger,self).on_train_end(logs)
        self.do_on_train_end()

  # define hyper parameters
 def do_after_training():
 	
        print("OPTMIZER CHANGED TO SECOND OPTIMIZER WHILE MONITORING NO LOSS VALIDATION DECREASE ") 
        model.compile(loss='categorical_crossentropy',  optimizer=ad1, metrics=['accuracy'])
              

min_delta =  0.000000000001


#https://stackoverflow.com/questions/47626254/changing-optimizer-in-keras-during-training
changer = OptimizerChanger(on_train_end= do_after_training, 
                           monitor='val_loss',
                           min_delta=min_delta,
                           patience=35,
                           verbose=2)

# After each epoch
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:

           print("MODEL  ACCURACY and Loss on TEST DATA AFTER EACH EPOCH =>") 
           results = model.evaluate(test_data,verbose=1)
           print("test loss, test acc:", results)

callback= myCallback()

####
# Create tensorboard callback (functionized because need to create a new one for each model)
import datetime
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
 
# Upload TensorBoard dev records
!tensorboard dev upload --logdir ./"tensorflow_hub" \
--name "My Oprimizer vs Typicall Optimizer" \
--description "Comparing My Optimizer with Typicall Optimizer Performance" \
--one_shot
##########################
##### Call above Functions
##########################
lr_sc = LearningRateScheduler(decay,verbose=1)

mycallback = [early_stopping,checkpoint
              ,reduce_lr
              ,callback
              ,lr_sc            
              ,create_tensorboard_callback(dir_name="tensorflow_hub", 
                                            experiment_name="cifar_exp_1")]  

mycallback = [reduce_lr,early_stopping,checkpoint]
