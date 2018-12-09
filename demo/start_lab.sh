# start jupyter notebook in this folder with the following cmd to use it remotely
jupyter lab --no-browser --ip=0.0.0.0  --port=$1

# In your local machine, type the follwing cmd to enable forwarding
# (which is aliased as ssh-b1-ipynb)
# ssh -X -N -f -L localhost:9100:localhost:9000 bvisionserver1.cs.unc.edu
