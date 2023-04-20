# Inspecting training logs:

### Via Tensorboard
To start logging on tensorboard, do:
```
tensorboard --logdir outputs
```
**NOTE:** If you want to view the tensorboard logs on the cluster you have three options:
1. Copy log directory to your local computer and view on there.

2. Start a tensorboard server with the same port that you use for jupyter lab. This requires that jupyter lab is not running, as you cannot have access to both. To start tensorboard to a specific port, write `tensorboard --logdir outputs --port=18475`. 

3. Use [tensorboard in VS Code](https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode)


### Plotting via jupyter notebook
Also, we included a notebook to export tensorboard logs to jupyter. See: ![](notebooks/plot_scalars.ipynb)