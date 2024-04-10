from typing import List
from torch import Tensor, randint
from torch.distributions.distribution import Distribution
from core import FunctionClass, ModifiedFunctionClass

class NoisyRegression(ModifiedFunctionClass):
    def __init__(
            self, 
            output_noise_distribution: Distribution,
            inner_function_class: FunctionClass,
        ):
        super(NoisyRegression, self).__init__(inner_function_class)
        self._out_noise_dist = output_noise_distribution

    def evaluate(self, x_batch: Tensor, params: List[Tensor] | Tensor) -> Tensor:
        y_batch = self._in_fc.evaluate(x_batch, params)
        y_batch_noisy = y_batch + self._out_noise_dist.sample()
        return y_batch_noisy

class ScaledRegression(ModifiedFunctionClass):
    def __init__(
            self,
            scale: float,
            inner_function_class: FunctionClass
        ):
        super(ScaledRegression, self).__init__(inner_function_class)
        self._scale = scale
    
    def evaluate(self, x_batch: Tensor, params: List[Tensor] | Tensor) -> Tensor:
        return self._scale * self._in_fc.evaluate(x_batch, params)


class Switching(FunctionClass):
    def __init__(self, inner_function_classes, switch_prob=1/10):
        #switch prob must be less than (n-1)/n where n is the number of function classes
        self.InnerFunctionClasses=inner_function_classes
        self.switch_prob=switch_prob
        self.n=len(inner_function_classes)
       
       
    def evaluate(self, x_batch, Tensor, params: List[Tensor] | Tensor) -> Tensor:
        switch=(torch.rand(x_batch) <self.n/((self.n-1)*self.switch_freq)).float() #generate indices for transition
        switch*=randint(1, self.n+1)
        
        
        
        for i in range(n):
            continue
        return 
            #generate
       
        #I want to sample a markow chain, where expected sojourn time is switch
        #should be equally likely to go to any of the others.
        #For speed reasons, we should probably only evaluate the necessary things  
   

class Multiple(FunctionClass):
    def __init__(self,sampling:Distribution ,inner_function_classes): #sampling is a distribution of the integers
        #0 to n-1, where n is the number of function classes.
       
        super(Multiple, self).__init__(*args)
        self.InnerFunctionClasses=inner_function_classes
        if sampling!=None:
            self.sampling=sampling #sample function. Defualt lambda : randint(n)
        else:
            self.sampling=lambda : randint(len(inner_function_classes))
       
    def evaluate(self, x_batch: Tensor, params: List[Tensor] | Tensor):
        return self.InnerFunctionClasses[self.sampling()].evalute(x_batch, params)
        
       
   
class Combination(ModifiedFunctionClass):

    def __init__(
            self,
            distribution:Distribution,
            inner_function_classes
        ):
        super(Multiple, self).__init__(*args)
        self.InnerFunctionClasses=inner_function_classes
        self.distribution=distribution #default standard 1/n of each.
        
    def evaluate(self, x_batch: Tensor, params: List[Tensor] | Tensor):
        if self.distribution!=None:
            weights=self.distribution
        else:
            weeights=[1/len(self.InnerFunctionClasses)]*len(self.InnerFunctionClasses)
        return sum([function_class.evaluate(x_batch, params)*weights[i] for i, function_class in enumerate(self.InnerFunctionClasses) ])
        
