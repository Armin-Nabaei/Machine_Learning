#Copyright [2023] [ARMIN NABAEI]

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
==============================================================================
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_ops
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras.optimizers.legacy import Optimizer 


class ArminAdam(Optimizer):
   
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=True,
                 name="ArminAdam", **kwargs):
        super(ArminAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon 
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
      for var in var_list:
          self.add_slot(var, 'm')
      for var in var_list:
          self.add_slot(var, 'v')
      if self.amsgrad:
          for var in var_list:
            self.add_slot(var, 'vhat')


    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        epsilon_t = 1e-8
        
        m_t = state_ops.assign(m,
        		   beta_1_t * m + (1.0 - beta_1_t) * grad,
        		   use_locking=self._use_locking)   
        		   beta_2_t * v + (1.0 - beta_2_t) * math_ops.square(grad),
        		   use_locking=self._use_locking)'''
        		   
        v_t = state_ops.assign(v,math_ops.square(1.0 - beta_2_t)*
                               ( v + beta_2_t * math_ops.square(grad)),use_locking=self._use_locking)
                               
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat, math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking) 
                                                                         
            vhat_t = vhat_t /  math_ops.square(1 - beta_2_t) 
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)
        var_t = math_ops.sub(var, lr_t * var_delta)
    
        var_update = state_ops.assign_sub(var, lr_t * var_delta,
                                          use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

            
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients[1 - beta_2_t]
                               use_locking=self._use_locking)
        v_t = state_ops.assign(v,math_ops.square(1.0 - coefficients['beta_2_t'])*( v + coefficients['beta_2_t'] ),
        		   use_locking=self._use_locking)

        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if not self.amsgrad:
            v_sqrt = math_ops.sqrt(v_t)
            var_update = state_ops.assign_sub(
                var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            v_hat_t = v_hat_t / math_ops.square(1 - beta_2_t) 
            with ops.control_dependencies([v_hat_t]):
              v_hat_t = state_ops.assign(
                  v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            var_update = state_ops.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(ArminAdam, self).set_weights(weights)
        
    def get_config(self):
        config = super(ArminAdam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config

