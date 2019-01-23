import torch
import numpy as np
import matplotlib.pyplot as plt
import inverse_utils
import dip_utils
import time


LR = 1e-4 # learning rate
MOM = 0.9 # momentum
NUM_ITER = 3000 # number iterations
WD = 1 # weight decay for l2-regularization
Z_NUM = 32 # input seed
NGF = 64 # number of filters per layer
nc = 1 #num channels in the net I/0
alpha = 1e-5 #learning rate of Lasso
alpha_tv = 1e-1 #TV parameter for net loss
LENGTH = 16384


CUDA = torch.cuda.is_available()
print("On GPU: ", CUDA)

if CUDA :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


save_loc = "/home/sravula/Projects/compsensing_dip-master/Results/" #The location you wish to save your results
test_type = "Denoising" #Imputation, CS, DCT, or Denoising
data_type = "Audio" #Audio or Chirp
sample = "clinton"  #If using audio, give the name of the sample you wish to use
std = 0.1 #the standard deviation of AWGN, if denoising


if data_type == "Audio":
    #Change the input to be the filename you choose
    wave_rate, wave_len, wave_res, nc, x0 = inverse_utils.read_wav("audio_data/" + sample + "_8192hz_2s.wav")

    if wave_len != LENGTH or nc > 1:
        print("ILL-FORMATTED WAV - TRY AGAIN")
        exit(0)

    x = inverse_utils.audio_normalise(x0, wave_res * 8)  #normalise the wave data to [-1,1]
elif data_type == "Chirp":
    #adjust f_start and f_end to generate chirp signals with different start and end frequencies
    f_start = 750
    f_end = 250
    x = inverse_utils.get_chirp(LENGTH, f_start, f_end)

    sample = "chirp:" + str(f_start) + "-" + str(f_end)

    inverse_utils.save_data(x, save_loc + test_type + "/" + sample + "/" + sample) #save the new wave
else:
    print("UNSUPPORTED DATA TYPE SELECTED - IF YOU WISH TO TEST AIR QUALITY DATA, PLEASE USE AirQuality_bench.py")
    exit(0)


if test_type == "Imputation" or test_type == "CS" or test_type == "DCT":
    num_measurements = [100, 500, 1000, 2000, 4000]
elif test_type == "Denoising":
    NUM_ITER = 300 #reduce iterations for denoising to prevent overfitting
    num_measurements = [LENGTH]
    noise = inverse_utils.get_noise(num_samples=LENGTH, nc = nc, std = std)
    x_orig = x.copy()
    x = x+noise
    inverse_utils.save_data(x, save_loc + test_type + "/" + sample + "/" + sample + "-noisy-" + str(std))  # save the wave again with noise added
else:
    print("UNSUPPORTED TEST TYPE. PLEASE CHOOSE: Imputation, CS, DCT, OR Denoising")
    exit(0)


error_dip = []
error_lasso = []
start = time.time()

i = 0
while i < len(num_measurements):

    if test_type != "Imputation": #if the test is not imputation, we do not get a list of kept samples
        phi, A = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=LENGTH)
    else:
        phi, A, kept_samples = inverse_utils.get_A(case=test_type, num_measurements=num_measurements[i], original_length=LENGTH)

    if test_type != "Denoising":
        inverse_utils.save_matrix(phi, save_loc + test_type + "/" + sample + "/" + sample + "-" + str(num_measurements[i])) #save the measurement matrix for later reference

    y = np.dot(phi, x)  #create the measurements

    num_instances = 5 #how many instances of lasso and DIP we wish to run to average results
    mse_lasso = 0
    mse_DIP = 0
    for t in range(num_instances):

        x_hat_lasso = inverse_utils.run_Lasso(A=A, y=y, output_size=LENGTH, alpha=alpha)

        if test_type == "Imputation": #for imputation, we only wish to calculate MSE on the imputed values
            imputed_samples = [z for z in range(0, LENGTH) if z not in kept_samples]
            mse_lasso = mse_lasso + np.mean((np.squeeze(x_hat_lasso)[imputed_samples] - np.squeeze(x)[imputed_samples])**2)
        elif test_type == "Denoising":
            mse_lasso = mse_lasso + np.mean((np.squeeze(x_hat_lasso) - np.squeeze(x_orig))**2)
        else:
            mse_lasso = mse_lasso + np.mean((np.squeeze(x_hat_lasso) - np.squeeze(x)) ** 2)

        x_hat_DIP = dip_utils.run_DIP(phi, y, dtype, NGF = NGF, LR=LR, MOM=MOM, WD=WD, output_size=LENGTH, num_measurements=num_measurements[i], CUDA=CUDA, num_iter=NUM_ITER, alpha_tv=alpha_tv)

        if test_type == "Imputation": #for imputation, we only wish to calculate MSE on the imputed values
            mse_DIP = mse_DIP + np.mean((np.squeeze(x_hat_DIP)[imputed_samples] - np.squeeze(x)[imputed_samples])**2)
        elif test_type == "Denoising":
            mse_DIP = mse_DIP + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x_orig))**2)
        else:
            mse_DIP = mse_DIP + np.mean((np.squeeze(x_hat_DIP) - np.squeeze(x)) ** 2)

    mse_lasso = mse_lasso/float(num_instances)
    mse_DIP = mse_DIP/float(num_instances)

    error_dip.append(mse_DIP)
    error_lasso.append(mse_lasso)
    print("\nNet MSE - " + str(num_measurements[i]) + " :", mse_DIP)
    print("Lasso MSE - " + str(num_measurements[i]) + " :", mse_lasso)

    i = i+1

end = time.time()
print("Execution Time: ", round(end-start, 2), "seconds")

DIP_results = np.zeros((len(num_measurements), 2))
Lasso_results = np.zeros((len(num_measurements), 2))

DIP_results[:,0] = num_measurements
Lasso_results[:,0] = num_measurements
DIP_results[:,1] = error_dip
Lasso_results[:,1] = error_lasso

#Save the results from testing - change save_loc and this block to customize your filenames
if test_type == "Denoising": #rename the test type for logging data
    inverse_utils.save_log(data=sample, test=test_type + "-" + str(std), method="DIP", results=DIP_results, filename=save_loc + test_type + "/" + sample + "/" + sample + "-" + "DIP" + ".txt")
    inverse_utils.save_log(data=sample, test=test_type + "-" + str(std), method="Lasso", results=Lasso_results, filename=save_loc + test_type + "/" + sample + "/" + sample + "-" + "Lasso" + ".txt")
else:
    inverse_utils.save_log(data=sample, test=test_type, method="DIP", results=DIP_results, filename=save_loc + test_type + "/" + sample + "/" + sample + "-" + "DIP" + ".txt")
    inverse_utils.save_log(data=sample, test=test_type, method="Lasso", results=Lasso_results, filename=save_loc + test_type + "/" + sample + "/" + sample + "-" + "Lasso" + ".txt")






