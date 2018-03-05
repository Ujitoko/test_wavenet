import numpy as np

from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = wavfile.read(path)[1][:, 0]

    data_ = normalize(data)
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins = np.linspace(-1, 1, 256)
    print(data_[0:-1].shape)
    print(data_[1::].shape)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # Encode targets as ints.
    targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]
    return inputs, targets

def one_hot_encoding(t_vec, num_classes):
    #t_oh = np.zeros((len(t_vec), num_classes)).astype(int)
    #t_oh[np.arange(len(t_vec)), t_vec] = 1
    #return t_oh
    targets = np.array(t_vec).reshape(-1)
    t_oh = np.eye(num_classes)[targets].astype("int")
    return (t_oh)

Selected_Acceleration_Label = {
    "G1SquaredAluminumMesh":0,
    "G2GraniteTypeVeneziano":1,
    "G3AluminumPlate":2,
    "G4Bamboo":3,
    "G5SolidRubberPlateVersion1":4,
    "G6Carpet":5,
    "G7FineFoamVersion2":6,
    "G8Cardboard":7,
    "G9Jeans":8,
    }    

class AccelerationDataset():
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        #self.files_z = [file for file in self.files if (file.find("_Z_") > 0) and (file.find("acc") > 0)]
        self.files_z = [file for file in self.files if (file.find("_Z_") > 0) and (file.find("acc") > 0) and (file.find("train1") > 0)]
        
    def len(self):
        return len(self.files_z)
        #return 10
    

    def getitem(self, idx):
        # wave
        file_name_acc_z = self.files_z[idx][:-7]
        pos_Z_ = file_name_acc_z.find("_Z_")

        file_name_vel_x = file_name_acc_z[:pos_Z_] + "_X_" + file_name_acc_z[pos_Z_ + len("_Z_"):]
        file_name_vel_y = file_name_acc_z[:pos_Z_] + "_Y_" + file_name_acc_z[pos_Z_ + len("_Z_"):]
        file_name_vel_mag = file_name_acc_z[:pos_Z_] + "_mag_" + file_name_acc_z[pos_Z_ + len("_Z_"):]

        z = np.load(os.path.join(self.root_dir, file_name_acc_z + "acc.npy")) * 2.0 - 1.0
        x = np.load(os.path.join(self.root_dir, file_name_vel_x + "vel.npy"))
        y = np.load(os.path.join(self.root_dir, file_name_vel_y + "vel.npy"))
        mag_vel = np.load(os.path.join(self.root_dir, file_name_vel_mag + "vel.npy"))

        #stacked = np.stack((x,y))
        #norm_vel = np.linalg.norm(np.stack((x,y)), axis=0)
        #print(norm_vel.shape)
        wave = np.stack((x,y,z,mag_vel))

        # label
        pos_label = file_name_acc_z.find("_Movement_")
        label_name = file_name_acc_z[:pos_label]
        label_num = Selected_Acceleration_Label[label_name]
        label_one_hot = one_hot_encoding(label_num, 9)
        
        sample = {"wave": wave, "label_num": label_num, "label_name": label_name, "label_one_hot": label_one_hot}
        return sample

    
    def getBatchTrain(self, is_random, time_step, batch_size):
        rand_batch_idx = np.random.randint(0, self.len(), size=batch_size)
        #print(rand_batch_idx)

        batch_input = []
        batch_output = []
        for idx in rand_batch_idx:
            item = self.getitem(idx)
            rand_time_start = 0#np.random.randint(0, item["wave"].shape[1] - time_step)
            #print(item["wave"].shape)
            wave_time_series = item["wave"][3,rand_time_start:rand_time_start+time_step][np.newaxis,:].transpose()
            one_hot = item["label_one_hot"]
            one_hot_time_series = np.repeat(one_hot, time_step, axis=0)

            input = np.concatenate((wave_time_series, one_hot_time_series), axis=1)
            #print(input.shape)
            batch_input.append(input)
            batch_output.append(item["wave"][2,rand_time_start+1: rand_time_start+time_step+1].transpose())
            
        batch_input_np = np.array(batch_input)
        batch_output_np = np.array(batch_output)
        return batch_input_np, batch_output_np#[:, np.newaxis]

    
    def getSequentialItem(self, time_step, batch_size=2):
        #rand_batch_idx = np.random.randint(0, self.len()-1, size=batch_size)
        #print(rand_batch_idx)
        batch_idx = np.arange(batch_size)
        
        batch_input = []
        batch_output = []

        item = self.getitem(0)
        time_series = item["wave"].shape[1]-batch_size
        #print(time_step)
        #print(batch_size)
        #print(time_series)
        for time in np.arange(batch_size):
            time_start = 0
            
            wave_time_series = item["wave"][3,time_start + time:time_start+time+time_step][np.newaxis,:].transpose()
            one_hot = item["label_one_hot"]
            one_hot_time_series = np.repeat(one_hot, time_step, axis=0)

            input = np.concatenate((wave_time_series, one_hot_time_series), axis=1)
            #print(input.shape)
            batch_input.append(input)
            batch_output.append(item["wave"][2,time_start+time+time_step].transpose())
            
        batch_input_np = np.array(batch_input)
        batch_output_np = np.array(batch_output)
        return batch_input_np, batch_output_np[:, np.newaxis]
    
def show_wave(wave, dirname, filename, y_lim=0):
    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)
    
    plt.figure(figsize=(30,10))
    plt.title('wave files', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if y_lim != 0:
        plt.ylim(0, y_lim)
    plt.plot(wave, color='r',linewidth=1.0)
    plt.savefig(os.path.join(dirname,filename + '.png'))
    plt.close()

def show_test_wav(waves, dirname, filename, y_lim=0):
    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)
    
    plt.figure(figsize=(30,10))
    plt.title('wave files', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if y_lim != 0:
        plt.ylim(0, y_lim)
    
    plt.plot(waves["test"], color='r',linewidth=1.0, label="test")
    plt.plot(waves["generated"], color='b', linewidth=1.0, label="generated")

    plt.legend()
    plt.savefig(os.path.join(dirname, filename + '.png'))
    plt.close()
