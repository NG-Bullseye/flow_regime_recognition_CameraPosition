import os
import shutil
def run(yaw_dir,dst_dir):
    # create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # iterate over all directories under yaw_dir
    for yaw_value_dir in os.listdir(yaw_dir):
        src_dir = os.path.join(yaw_dir, yaw_value_dir)

        # iterate over all directories under each yaw directory
        for training_set_dir in os.listdir(src_dir):
            training_dir = os.path.join(src_dir, training_set_dir)

            # make sure it's a directory
            if os.path.isdir(training_dir):
                # generate new directory name
                new_dir_name = "{}_{}_training_output".format(yaw_value_dir, training_set_dir)

                # create full path for destination directory
                dst_full_dir = os.path.join(dst_dir, new_dir_name)

                # copy entire directory tree to new location
                shutil.copytree(training_dir, dst_full_dir)
