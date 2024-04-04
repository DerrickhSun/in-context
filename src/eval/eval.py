
#big picture. Want to be able to evaluate differetn models and get data on a lot of benchmarks. 
import numpy as np
import torch
from function_classes.wrappers import NoisyRegression, ScaledRegression
    

#questions for nelson. 
    #exactly what information can I access in the contextmodel?
    #exactly what information is typically included in config data

# runs all eval
# config_data needs function_class and accuracy_func
# may accept test_size, generation distribution
def basic_eval(model, config_data):
    function_class=config_data.get("function_class")
    accuracy_func=config_data.get("accuracy_func") #could also use an acc func instead. #has to do each cell individually
    samples = config_data.get("samples", 1000)

    batch_size = function_class.batch_size
    seq_length = function_class.sequence_length

    #create thing
    acc=torch.zeros((samples, batch_size, seq_length))

    for i, (x_batch, y_batch) in zip(range(samples), function_class):
        output = model(x_batch, y_batch)
        acc[i] = accuracy_func(output, y_batch)
    
    acc=torch.reshape(acc, (samples*batch_size, seq_length))
    std=torch.std(acc, dim=0)
    stats={"accuracy": acc.mean(dim=0), "std": std, "std_mean": std/np.sqrt(samples*batch_size)}
    quantiles=torch.quantile(acc, torch.Tensor([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]), dim=0)
    stats["max"]=quantiles[len(quantiles)-1]
    stats["min"]=quantiles[0]
    for i in range(1, len(quantiles)-1):
        stats["quantile"+str([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1][i])]=quantiles[i]
    return stats
        
def robustness_main_task(model, config_data):
    #do the robustness validations. Adding noise, etc. 


    robustness_tasks=[]

    robustness_nums={}

    for scale in [0.125, 0.25, 0.5, 2, 4, 8]:
        robustness_tasks.append(["scaled_x", scale])
        robustness_tasks.append(["scaled_y", scale])
        
    for noise in [0.0625, 0.125, 0.25, 0.5, 1]:
        robustness_tasks.append(["noise_x", noise])
        robustness_tasks.append(["noise_y", noise])    

    #might add some other distributionsional shifts or change these later. 
    
    function_class=config_data.get("function_class")
    accuracy_func=config_data.get("accuracy_func") #could also use an acc func instead
    samples=config_data.get("samples", 1000)
    
    noise_x_func=config_data.get("noise_x_func") #function that takes an array and a scaling factor, and outputs
    noise_y_func=config_data.get("noise_y_func") #noised variables. Should be main task specific
    
    #scaled_x_tasks=[0.125, 0.25, 0.5, 2, 4,  8] #can do it this way instead. Might be better, but would require some extra code some places
    #scaled_y_tasks=[0.125, 0.25, 0.5, 2, 4,  8]
    #noise_x_tasks=[0.0625, 0.125, 0.25, 0.5, 1]
    #noise_y_tasks=[0.0625, 0.125, 0.25, 0.5, 1]
    #function_classes=[]
    #for scale in scaled_x_tasks:
    #    function_classes.append    #must implement the correct function class in wrappers first
    #for scale in scaled_y_tasks:
    #    function_classes.append(ScaledRegression(inner_function_class=function_class, scale=scale)) 
    #for noise in noise_x_tasks>
    #   function_classes.append #must implement the correct function class in wrappers first
    #for noise in noise_y_tasks:
    #    function_classes.append(NoisyRegression(inner_function_class=function_class, output_noise_distribution=noise_y_func))

    batch_size = function_class.batch_size
    seq_length = function_class.sequence_length

    for task, j in enumerate(robustness_tasks):
        
        acc=torch.zeros((samples, batch_size, seq_length))
    
        for i, (x_batch, y_batch) in zip(range(samples), function_class):

            curxs=x_batch
            curys=y_batch
            
            if task[0]=="scaled_x":
                curxs*=task[1]
            elif task[0]=="scaled_y":
                curys*=task[1]
            elif task[0]=="noise_x":
                curxs=noise_x_func(task[1])(curxs)
            elif task[0]=="noise_y":
                curys=noise_y_func(task[1])(curys)
                
            output = model(curxs, curys)
        
            robustness_nums[i] = accuracy_func(output, curys)

        acc=torch.reshape(acc, (samples*batch_size, seq_length))
        std=torch.std(acc, dim=0)
        robustness_nums[task[0]+"_"+str(task[1])+"_accuracy"]=acc.mean(dim=0)
        robustness_nums[task[0]+"_"+str(task[1])+"_std"]=std
        robustness_nums[task[0]+"_"+str(task[1])+"_std_mean"]=std/np.sqrt(samples*batch_size)

        quantiles=torch.quantile(acc, torch.Tensor([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]), dim=0)
        robustness_nums[task[0]+"_"+str(task[1])+"_max"]=quantiles[len(quantiles)-1]
        robustness_nums[task[0]+"_"+str(task[1])+"_min"]=quantiles[0]
        
        for i in range(1, len(quantiles)-1):
            robustness_nums[task[0]+"_"+str(task[1])+"quantile"+str([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1][i])]=quantiles[i]
        

    return robustness_nums
    
    
def expressivity_main_task(model, config_data):
    
    #what natural task are there without accessing inner information. Like its easy with specific cases. 
    #if it has a latent variable, we can increase that.
    #otherwise quite hard.
    
    return {}

    
def performance_eval(model, config_data): #input is a trained model. 
    #config_data should have all necessary data to test a model in real time at typical problems. 
    #should we include more specific "wierd" data? Probably should not impact performance mostly
    performance_data={}
    
    return performance_data #ideally a dictionary
    
    
    
    #Should evaluate the model at the function class. 
    #returns score generated by function class (or should loss function in general be separate)?
    
def robustness_general_task(model, config_data):
    
    robustness_nums={}
    
    return nobustness_nums

    
def eval_model(model, config_data):

    
    
    
    #input should only be a basic model archiecture, maybe also a category.
    #could also take a trained model, but in that case only run main_model benchmarks
    
    
    
    
    #some way to select the benchmarks we want. Should incorporate different categories:
        #main task accuracy
        #robustness. How should this be incorporated
        #some sort of generalizability/expressivity
    
    #might be necessary to train multiple models based on same architecture.
    #probably ned to restrict config
    
    #train main model here. Used for most stuff.
    #not certain about exact syntax. 
    
    only_main_model=config_data.get("only_main_model") or config_data.get("pretrained")
    
    
    if config_data.get("pretrained")==False:
        #train main model.
        pass
    else:
         main_model=model
    
    benchmark_nums={}
    
    benchmark_nums.update(basic_eval(main_model, config_data))
    
    benchmark_nums.update(robustness_main_task(main_model, config_data))
    
    benchmark_nums.update(expressivity_main_task(main_model, config_data))

    benchmarks_nums.update(performance_eval(main_model, config_data))
    
    if only_main_model: 
        #Fill in the relevant slots with NAN. 
        pass
        
    else:#these require additional training to get evaluations. 
        
        benchmark_nums.update(robustness_general_task(model, config_data))
    
        benchmark_nums.update(expressivity_general_task(model, config_data))
    
    
    return benchmark_nums


