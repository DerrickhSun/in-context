
#big picture. Want to be able to evaluate differetn models and get data on a lot of benchmarks. 

    
    

#questions for nelson. 
    #exactly what information can I access in the contextmodel?
    #exactly what information is typically included in config data

import numpy as np




def basic_eval(trained_model, config_data):
    function_class=config_data.get("function_class")
    accuracy_func=config_data.get("accuracy_func") #could also use an acc func instead
    samples=config_data.get("samples") #change this if it is not correct
    acc=np.zeros(samples)
    for i in range(samples):
        seq_batch = [[(x, y) for x, y in function] for function in function_class.get_function_iter()]
        xs = [[xy_pair[0] for xy_pair in seq] for seq in seq_batch]
        ys = [[xy_pair[1] for xy_pair in seq] for seq in seq_batch] #change if they write something better
        output = model(xs, ys)
        acc[i] = accuracy_func(output, ys)
    return {"accuracy": acc.mean()}
        
def robustness_main_task(trained_model, config_data):
    
    
    #do the robustness validations. Adding noise, etc. 
    robustness_tasks=["scaled_x2", "scaled_x4", "scaled_x8", "scaled_y2", "scaled_y4", "scaled_y8", "noise_x.0625", "noise_x.25", "noise_x1", "noise_y.0625", "noise_y.25", "noise_y1"]
    
    #might add some other distributionsional shifts or change these later. 
    
    function_class=config_data.get("function_class")
    accuracy_func=config_data.get("accuracy_func") #could also use an acc func instead
    samples=config_data.get("samples") #change this if it is not correct
    
    noise_x_func=config_data.get("noise_x_func") #function that takes an array and a scaling factor, and outputs
    noise_y_func=config_data.get("noise_y_func") #noised variables. Should be main task specific
    
    #must have some check if task is classification rather than simply predicting a value. E.G. randomly flipping
    #maybe ask for implementation of this by the other guys 
    #does it make more sense to do everything here locally?
    
    #function_classes=None #Might add this if it makes more sense to move stuff to the backend. 
    
    for i in range(samples):
        seq_batch = [[(x, y) for x, y in function] for function in function_class.get_function_iter()]
        xs = [[xy_pair[0] for xy_pair in seq] for seq in seq_batch]
        ys = [[xy_pair[1] for xy_pair in seq] for seq in seq_batch] #change if they write something better
        
        for task, i in enumerate(robustness_tasks):
            curxs=xs
            curys=ys
            
            if task[:8]=="scaled_x":
                curxs=xs*int(task[8])
            elif task[:8]=="scaled_y":
                curys=ys*int(task[8])
            elif task[:7]=="noise_x":
                curxs=noise_x_func(xs, task[7])
            elif task[:7]=="noise_y":
                curys=noise_y_func(ys, task[7])
                
            output = model(curxs, curys)
            robustness_nums[task] = accuracy_func(output, curys)
    
    return robustness_nums
    
    
def expressivity_main_task(trained_model, config_data):
    
    #what natural task are there without accessing inner information. Like its easy with specific cases. 
    #if it has a latent variable, we can increase that.
    #otherwise quite hard.
    
    return {}

    
def performance_eval(trained_model, config_data): #input is a trained model. 
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



