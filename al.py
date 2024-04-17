from Algorithm_RIS import generate_channel

#Converting this to python: 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

def statistical_model():

    #f = 2.4*10^9;
    #M = 1;                    # Antennas in TX
    #N = 128;                    # Antennas in RIS
    #tau =10000;                   # Time slots
    #risset = 0;

    f = 2.4*10**9 # frequency in Hz
    M = 1 # Antennas in TX
    N = 128 # Antennas in RIS
    tau = 10000 # Time slots
    risset = 0 # RIS setting, which means no RIS in this case

    #Tx_pos = [3 0 0];
    #RIS_pos = [0 -3 0];
    #distance = 3;
    #theta_real = 20;
    #deltaX_r = distance * cosd(theta_real);
    #deltaY_r = distance * sind(theta_real);
    #obj_pos = [RIS_pos(1) + deltaX_r, RIS_pos(2) + deltaY_r,0];

    TX_pos = np.array([3, 0, 0]) # Stationary Tx position

    RIS_pos = np.array([0, 3, 0]) # Stationary RIS position

    distance = 3

    theta_real = 20


    # calculate deltaX_r using trigonometry, where deltaX_r means the x-coordinate of the object position relative to the RIS position
    # deltaX_r is the x-coordinate of the object position relative to the RIS position
    deltaX_r = distance * np.cos(np.deg2rad(theta_real)) 

    # calculate deltaY_r using trigonometry, where deltaY_r means the y-coordinate of the object position relative to the RIS position
    # deltaY_r is the y-coordinate of the object position relative to the RIS position
    deltaY_r = distance * np.sin(np.deg2rad(theta_real))

    # object position
    obj_pos = [RIS_pos[0] + deltaX_r, RIS_pos[1] + deltaY_r, 0]

    #snrdb = 10;
    #x = ones(M,tau)./sqrt(M);
    #snr_deci = db2pow(snrdb);      % SNR in linear scale
    #%array_gain =10*log10(N^8);
    #npower = 1;               % noise power is 1
    #Tx_p = db2pow(220);
    #spower = npower*snr_deci*Tx_p;      % sig power
    #samp = sqrt(spower);    % signal amplitude in each channel
    #x = samp *x;

    # SNR in dB
    snrdb = 10

    # This line has the functionality of creating an input signal X with M antennas and tau time slots, where each element is 1/sqrt(M) to normalize the power.
    X = np.ones((M, tau)) / np.sqrt(M) 
    snr_deci = 10**(snrdb / 10)  # SNR in linear scale
    npower = 1  # noise power is 1
    Tx_p = 10**(220 / 10)  # Tx power in linear scale
    spower = npower * snr_deci * Tx_p  # signal power
    samp = np.sqrt(spower)  # signal amplitude in each channel
    X = samp * X  # apply signal amplitude to input signal


    #[H_0,~,H_3h] = generate_channel(Tx_pos,RIS_pos,obj_pos,M, N,f);
    #% [~,~,H_3h] = generate_channel_RIS_weights(Tx_pos,RIS_pos,obj_pos,M, N,f,diag(w_cont));

    # This line has the purpose of generating the channel matrices H_0 and H_3h with the purpose of evaluating the received signal y1 and y0.
    # It evaluates the channel matrices based on the given positions of the transmitter, receiver, and object, as well as the frequency
    [H_0,_, H_3h] = generate_channel(TX_pos, RIS_pos, obj_pos, M, N, f)

    #s1 = H_3h * x;
    #n1 = (randn(M,tau)+1i*randn(M,tau))./sqrt(2);  % noise
    #y1 = transpose(s1 + n1);

    s1 = H_3h @ X
    n1 = (np.random.randn(M, tau) + 1j * np.random.randn(M, tau)) / np.sqrt(2)  # noise
    y1 = np.transpose(s1 + n1) # This is the received signal with the 3-hop channel with transmitter-RIS-object

    #s0 = H_0 * x;
    #n0 = (randn(M,tau)+1i*randn(M,tau))./sqrt(2);  % noise
    #y0 = transpose(n0);

    print(H_0.shape) # (128, 128)
    print(X.shape) # (1, 10000)

    H_0 = np.array([128, 128])

    X = np.array([1,1000])

    s0 = np.dot(H_0, X)
    n0 = (np.random.randn(M, tau) + 1j * np.random.randn(M, tau)) / np.sqrt(2) # noise
    y0 = np.transpose(n0) # This is the received signal with the direct LOS channel, which does not include the object

    #%% statistic 
    #% Create Histogram of Outputs
    #h1a = abs(y1);
    #h0a = abs(y0);
    #thresh_low = min([h1a;h0a]);
    #thresh_hi  = max([h1a;h0a]);
    #nbins = 100;
    #binedges = linspace(thresh_low,thresh_hi,nbins);
    #figure;
    #histogram(h0a,binedges)
    #hold on
    #histogram(h1a,binedges)
    #hold off
    #title('Target-Absent Vs Target-Present Histograms')
    #legend('Target Absent','Target Present')

    # Finding the magnitudes of received signals, and
    # create histograms to visualize target present vs target absent. 
    # We do this by find the probabilities of detection and false alarm by sweeping threshold

    h1a = np.abs(y1) # Magnitudes of received signal with 3-hop channel, which includes the object. 
    h0a = np.abs(y0) # Magnitudes of received signal with direct LOS channel, which does not include the object.
    thresh_low = min(np.concatenate((h1a, h0a)))
    thresh_hi = max(np.concatenate((h1a, h0a)))
    nbins = 100
    binedges = np.linspace(thresh_low, thresh_hi, nbins).flatten()
    fig, ax = plt.subplots()
    ax.hist(h0a, bins=binedges, alpha=0.7, label='Target Absent', rwidth = 0.85)
    ax.hist(h1a, bins=binedges, alpha=0.7, label='Target Present', rwidth = 0.85)
    ax.set_title('Target-Absent Vs Target-Present Histograms')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Save figure 
    img_hist = BytesIO()

    # Save figure for the html main page
    plt.savefig(img_hist, format='png')

    # Return histogram image
    # Before encoding, we start at the beginning of the data buffer
    img_hist.seek(0)

    # Encode image as base64
    plot_url2 = base64.b64encode(img_hist.read()).decode('utf-8')

    #plot_url = base64.b64encode(img.getvalue()).decode()

    # Close figure
    img_hist.close()    


    #nbins = 1000;
    #thresh_steps = linspace(thresh_low,thresh_hi,nbins);
    #sim_pd = zeros(1,nbins);
    #sim_pfa = zeros(1,nbins);
    #for k = 1:nbins
    #    thresh = thresh_steps(k);
    #    sim_pd(k) = sum(h1a >= thresh);
    #    sim_pfa(k) = sum(h0a >= thresh);
    #end
    #sim_pd = sim_pd/tau;
    #sim_pfa = sim_pfa/tau;
    #pfa_diff = diff(sim_pfa);
    #idx = (pfa_diff == 0);
    #sim_pfa(idx) = [];
    #sim_pd(idx) = [];
    #minpfa = 1e-6;

    # Plot ROC based on probabilities of detection vs false alarm. 
    nbins = 1000
    thresh_steps = np.linspace(thresh_low, thresh_hi, nbins) # 
    sim_pd = np.zeros(nbins) # Initialize array for probabilities of detection
    sim_pfa = np.zeros(nbins) # Initialize array for storing the probability false alarm values
    for k in range(nbins):
        thresh = thresh_steps[k]
        sim_pd[k] = np.sum(h1a >= thresh) 
        sim_pfa[k] = np.sum(h0a >= thresh)

    sim_pd = sim_pd / tau # Normalize probabilities by number of time slots
    sim_pfa = sim_pfa / tau # Normalize probabilities by number of time slots
    pfa_diff = np.diff(sim_pfa) # Find difference between consecutive pfa values
    idx = np.concatenate(([False], pfa_diff == 0)) # Find indices where difference is 0 by concatenating False with the difference 0 boolean mask
    sim_pfa = sim_pfa[~idx] # Replace pfa values at indices where difference is 0 with the values of sim_pfa where idx is False
    sim_pd = sim_pd[~idx] # Remove corresponding pd values
    minpfa = 1e-6 # Set minimum pfa threshold



    #Nu = sum(sim_pfa >= minpfa);
    #sim_pfa = fliplr(sim_pfa(1:Nu)).';
    #sim_pd = fliplr(sim_pd(1:Nu)).';
    #% [theor_pd,theor_pfa] = rocsnr(snrdb,'SignalType',...
    #%     'NonfluctuatingNoncoherent',...
    #%     'MinPfa',minpfa,'NumPoints',Nu,'NumPulses',1);
    #% semilogx(theor_pfa,theor_pd)
    #% hold on
    #figure;
    #semilogx(sim_pfa,sim_pd,'r.')
    #title('Simulated ROC Curves')
    #xlabel('Pfa') % The probability of false alarm
    #ylabel('Pd') % The probability of detection
    #grid on
    #legend('Simulated','Location','SE')

    Nu = np.sum(sim_pfa >= minpfa)
    sim_pfa = np.flip(sim_pfa[:Nu]) # Flip arrays since semilogx plots from right to left
    sim_pd = np.flip(sim_pd[:Nu]) # Flip arrays since semilogx plots from right to left
    fig, ax = plt.subplots()
    ax.semilogx(sim_pfa, sim_pd, 'r.')
    ax.set_title('Simulated ROC Curves')
    ax.set_xlabel('Pfa')  # The probability of false alarm
    ax.set_ylabel('Pd')  # The probability of detection
    ax.grid(True)
    ax.legend(['Simulated'], loc='lower right')
    #plt.show()
    

    # Save figure as png 
    #img_hist = BytesIO()
    img_ROC = BytesIO()

    # Save figure for the html main page
    #plt.savefig(img_hist, format='png')

    plt.savefig(img_ROC, format='png')

    # Before encoding, we start at the beginning of the data buffer
    #img_hist.seek(0)
    img_ROC.seek(0)

    # Encode image as base64
    #plot_url2 = base64.b64encode(img_hist.read()).decode('utf-8')
    plot_url3 = base64.b64encode(img_ROC.read()).decode('utf-8')


    # Close figure
    #img_hist.close()
    img_ROC.close()

    # Return plot urls
    return f"data:image/png;base64,{plot_url2}", f"data:image/png;base64,{plot_url3}"
