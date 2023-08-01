# Diplomarbeit LEONARD-RICCARDO HANS WECKE
##Installation
virtual enviroment is: env_python3.7_tens2.3_dev (even though its tensorflow2.4.0)
Tensorflow 2.4.0
Python 3.7
np 11.8.5 (oder so)
requirements.txt
CPU support
##Configuration
param.yaml input output pfade angeben
##Image preprocessing
56x56px , greyscale
##Image labeling and balancing
##Image Sorting
Sort into folders by yaw values
##Training
training for each yaw value folder
##Evaluation
Accuracy eval and performance metrics
##gradcam++
in jupiternotebook files 
##Custom Baysian optimization for simulation of camera positioning
use mapping of yaw and acc for simulation of camera movement
Benchmarkes CBO to Gridsearch, Randomsearch, BO and other optimization
## Postgres DB
# setup 
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
which psql
export PATH=$PATH:pathto/pgsql/bin
close consol maybe to take effect

sudo -u postgres psql

[sudo] Passwort für lwecke: 
psql (12.15 (Ubuntu 12.15-0ubuntu0.20.04.1))
Type "help" for help.

postgres=# \password
Enter new password for user "postgres": 
Enter it again: 
postgres=# \q (quit the consol)
postgres=# \l (list the databases)

dann verbinden mit database setup
# connect 
Host: localhost
Port: 5432 (default PostgreSQL port)
User: postgres
Password: 
## Execution

1 Preprocessing the images
2 Bulk Train
3 postprocess gradcam_mean
4 postprocess recommendation scalar from gradcam_mean images for each yaw value
5 Use BlackboxOptimisation project to load the generated yaw_acc.csv and yaw_recScalar.csv for optimization simulation
