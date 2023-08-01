import os
from modules.Utility.database import Sim_data_db as sim_db
if __name__ == '__main__':
    db = sim_db('postgres',)
    # Add data to the database
    db.put(yaw=45.1,image_path='\home\leo\succmd51.png')
    db.put(yaw=45.1,gradcam_mean_path='\home\leo\gradcams\succmd_gradcam51.png')
    db.put(yaw=45.1,rec_scalar=-1.54)
    db.put(yaw=45.1,acc=0.55)

    db.put(yaw=4,image_path='\home\leo\succmd4.png')
    db.put(yaw=4,gradcam_mean_path='\home\leo\gradcams\succmd_gradcam4.png')
    db.put(yaw=4,rec_scalar=-1.41)
    db.put(yaw=4,acc=0.430)

    db.put(yaw=3,image_path='\home\leo\succmd3.png')
    db.put(yaw=3,gradcam_mean_path='\home\leo\gradcams\succmd_gradcam3.png')
    db.put(yaw=3,rec_scalar=1.31)
    db.put(yaw=3,acc=0.33)
    # Get data from the database
    print(db.get('gradcam_mean_path', 'yaw', 45.1))

    # Don't forget to close the connection when you're done
    db.close()