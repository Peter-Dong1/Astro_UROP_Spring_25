# Astrophysics Meeting Notes Total

Course: Astrophysics Research Kavli (https://www.notion.so/Astrophysics-Research-Kavli-16a475833b3280849309f08c8d11cc6a?pvs=21)
Resource Type: Notes
Semester: Fall 2025
Tags: Physics: Astrophysics

top 10k - narrow scope

python bexvar_histograms.py "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/640/processedbatch/feature.pkl" --outdir "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/640features”

10-20 lc per cluster

Ideally more than this → less clusters

# Task List:

- [ ]  Active learning - rerank isolation forest (try to get light curves simliar to the ones that we have)
    - [ ]  Look for where the features are
    - [ ]  print the outputs
- [x]  Inside each different cluster, do PCA to see the groups inside of the cluster
- [x]  Fix cluster significance
    - [x]  Calculate significance for entire dataset then calculate the difference between any given cluster and massive dataset
- [ ]  Fix corner plots
    - [x]  Try log plots
    - [ ]  remove really big outliers
- [ ]  Fix PCA of features
    - [x]  Try log plots
    - [ ]  Plot nearest neighbors to the outlier curves
- [x]  Add more features
- [ ]  HDBSCAN instead of DBSCAN for UMAP

301: 99.7

302: 95

303: 68

304: 40

305: 30

306: 20

310: 30 without exvar

311: 20 w/o exvar - 50 min cluster size, 25 min samples  

Topics:

- Haven’t really done anything since last meeting - swamped with amazon work
- Future work:
    - Fall? - Likely don’t want to continue in the fall → perhaps can pick back up IAP or something
    - What should i do?
        - Write up document about everything that I’ve found and what I’ve tried
        - etc
- 

260:

- min_cluster size = 20

265 - 

Min cluster size 25

12 - min sample

266 - 50, 25

267 - 100, 50

python sample_clusters.py --run 266 --samples 25 --outdir "/home/pdong/Astro UROP/z New Feature Extraction Pipeline/plots/all266/CLUSTERS”

231:

```python
DEFAULT_MIN_CLUSTER_SIZE = 5 # Smaller
DEFAULT_EPSILON = 0.13
DEFAULT_EOM = 'eom'
DEFAULT_MIN_SAMPLES = 3

2nd pass:
divide nums by 1.5
divide eps by 3
leaf
```

232

```jsx
DEFAULT_MIN_CLUSTER_SIZE = 7 # Smaller
DEFAULT_EPSILON = 0.11
DEFAULT_EOM = 'eom'
DEFAULT_MIN_SAMPLES = 5

2nd pass:
divide nums by 1.5
divide eps by 3
leaf
```

233 - 232, but leaf

234 - 231 correctly

235 - 232

236 - 233

237 - 233 + leaf and epsilon 0.2

240 - 231

241 - 232

242 - 233

243 - 233 + leaf and epsilon 0.2

200: 

- min cluster: 3
- epsilon: 0.5
- EOM: leaf
- min samples: 3

201: 

- min cluster: 3
- epsilon: 0
- EOM: leaf
- min samples: 3

202: 

- min cluster: 3
- epsilon: 0
- EOM: EOM
- min samples: 3

211: 

- min cluster: 3
- epsilon: 0
- EOM: leaf
- min samples: 3

212: 

- min cluster: 3
- epsilon: 0
- EOM: eom
- min samples: 3

213: 

- min cluster: 3
- epsilon: 0.5
- EOM: leaf
- min samples: 3

214: 

- min cluster: 3
- epsilon: 0.3
- EOM: leaf
- min samples: 3

215: 

- min cluster: 3
- epsilon: 0.1
- EOM: leaf
- min samples: 3

216: 

- min cluster: 3
- epsilon: 0.05
- EOM: leaf
- min samples: 3

217: RERUN OF 215

- min cluster: 3
- epsilon: 0.1
- EOM: leaf
- min samples: 3

218: 

- min cluster: 3
- epsilon: 0.08
- EOM: leaf
- min samples: 3

219: 

- min cluster: 3
- epsilon: 0.15
- EOM: leaf
- min samples: 3

220: 

- min cluster: 5
- epsilon: 0.10
- EOM: leaf
- min samples: 3

221: 

- min cluster: 5
- epsilon: 0.15
- EOM: leaf
- min samples: 3

222: 

- min cluster: 5
- epsilon: 0.13
- EOM: leaf
- min samples: 3

223: 

- min cluster: 5
- epsilon: 0.13
- EOM: leaf
- min samples: 3
- Clipped

Distribution of excess variance of all light curevs

train random forest from features to cluster

- Can see feature importnace
- Go cosine similarlity list → how many different clusters they come into

100: 5 min cluster, 0 epsilon

101: 5 min cluster, 1 epsilon

103: 5 min cluster, 5 epsilon

106: 3 min cluster, 0 epsilon

107: 3 min cluster, 3 epsilon

108: 3 min cluster, 5 epsilon

Notes:

- Implemented cosine similarty → had to use normalize the feature vectors to unit vectors, then compute euclidean distsance
    - This is because cosine didn’t work with HDBSCAN for somre reason, but **||x - y||² = 2(1 - cos(x,y))**

ON A COMPUTE NODE:

ssh -L 8000:localhost:8000 [pdong@orcd-login001.mit.edu](mailto:pdong@orcd-login001.mit.edu)

salloc -N 1 -n 4 -p mit_normal

uvicorn app:app --reload --host 0.0.0.0 --port 8000

Run in ssh under webapp folder:
uvicorn app:app --reload

Run in local terminal for webapp:

ssh -L 8000:localhost:8000 pdong@orcd-login001

focus on one thing

- Try on all sources with teh statisical feature extraction and clustering
- Try to tune to work better
- Cross link the Isolation forest and the clustering algorithms
- keep track of average statistics between each of the cluster
- matplotlib & Plotly to show bettter clusters
    - or each one of them
- Cosine similarity and rank top 100 clusters
- Add back mag ratio
- FIX VISUALIZATION - TOP PRIORITY
    - Add max amplitude

UROP PROGRESS
- Started to transfer things into two seperate files so i can just load features instead of caluclating them every time

- 

Distance metric → cosine similarity/inner product idk:

- Gives you a ranked list of silmilarity
- Can get the difference to a single object
- Create folders of plots for cosine similarty ranking for each of the 3 intersting light curves

Is there a way to weigh excess_var more than the rest

- see if this is possible

rsync -ah -rltpDvpz -e 'ssh -l pdong2' /pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned [data.bridges2.psc.edu](http://data.bridges2.psc.edu/):../../../ocean/projects/phy240105p/pdong2

rsync -ah -rltpDvpz -e 'ssh -l pdong2' /pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned [data.bridges2.psc.edu](http://data.bridges2.psc.edu/):../../../ocean/projects/phy240105p/pdong2

rsync -ah -rltpDvpz -e 'ssh -l pdong' [pdong@orcd-login001.](mailto:pdong@orcd-login001.rcac.purdue.edu)mit.edu:/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned /ocean/projects/phy240105/pdong2/

rsync -ah -rltpDvpz -e 'ssh -l pdong' [pdong@orcd-login001.](mailto:pdong@orcd-login001.rcac.purdue.edu)mit.edu:/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned /ocean/projects/phy240105p/pdong2/

potential error: processed_light_curves needs to be deepcopied

# 5/2

### Accomplished:

- Added isolation forest reweighting
    - Try Mahalanobis distance instead of euclidean
    - Unsure if euclidean distance works properly
- Tried out some more features
    - Want to figure out how to use Baxvar
    - 

### Questions:

### Notes:

# 4/23

## Meeting notes:

### Accomplished:

- PCA on each of the groups inside of a cluster
- Cluster Significance is correctly calculated
    - D
- Log plots on big PCA and Corner Plots
- New features
    - FIND FEATURES

chat

make KD version of the corner plot

### Questions:

## Notes:

Symmetric Error - works

ExcessVariance - add

Magnitude only - does not work with dataset

Transfer files on the way over: ADD Z for compression through bandwith

rsync command:

rsync -ah -rltpDvp -e 'ssh -l pdong2' /pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned [data.bridges2.psc.edu](http://data.bridges2.psc.edu/):../../../ocean/projects/phy240040p/pdong2/light_curve_files

rsync -rltpDvp -e 'ssh -l PSC-username' [data.bridges2.psc.edu](http://data.bridges2.psc.edu/):source_directory  target_directory

rsync -ah -rltpDvp -e 'ssh -l pdong' [pdong@orcd-login001.](mailto:pdong@orcd-login001.rcac.purdue.edu)mit.edu:/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned /ocean/projects/phy240040p/pdong2/light_curve_files

4/19 - 1.3 hrs

# 4/18

### Accomplished:

Made corner plot better

- Remove grey clusters from corner plot

Created file that shows what percent feature distinguished it

Added the special light curves into the cluster

- They’re always showing up in the noise cluster, which i guess makes sense
- On the search for some more features that can help me bias towards higher magnitude events or something

Moving on to transformer model - I think i got time parameter working properly, but waiting for it to train to see if any difference has been made

### Questions:

- Go through light curves
- Could i get some help working with the new cluster - Can only train one model at a time
- Want to be able to do multiple
    - Help from other student doing it?

### Notes:

TODO:
Active learning - rerank the isolation forest

- look for where the features
- Print them so that i can see the outputs

PCA inside the different cluster

CLUSTER SIGNIFICANCE IS NOT WORKING

- CALCULATE SIGNIFICANCE FOR THE ENTIRE DATASET THEN CALCULATE THE DIFFERENCE BETWEEN ANY GIVEN CLUSTER AND THE MASSIVE DATASET
- 

Probably use log plot for each of the corner plots

Calculate objects with similar scores to that object and try to cluster them

- Metric of similarlity
- Look at where the land in latent space and check nearest neighbors

Problem space has changed

- Work with the fact that we had done it before

remove Highest points and calculate difference

# 4/10

What I’ve Accomplished:

Questions:

Notes:

Remove grey clusters from corner plot

- See if i can color code them any better
- See if i can mark each cluster with the feature that distinguished it (maybe like a percent)

Active learning - rerank the isolation forest

- look for where the features
- Print them so that i can see the outputs
- 

# 4/4

Objectives:

- ~~Set up github~~
- Port all data over - IP
- Set up Virtual env - IP
- Add some features to the feature extraction code
- Transformer model
    - Get time working
    - Train a new one
    - Modernize the code
- RNN model
    - Get Log-Liklihood working

rsync command:

rsync -ah -rltpDvp -e 'ssh -l pdong2' /pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned [data.bridges2.psc.edu](http://data.bridges2.psc.edu/):../../../ocean/projects/phy240040p/pdong2/light_curve_files

rsync -rltpDvp -e 'ssh -l PSC-username' [data.bridges2.psc.edu](http://data.bridges2.psc.edu/):source_directory  target_directory

rsync -ah -rltpDvp -e 'ssh -l pdong' [pdong@orcd-login001.](mailto:pdong@orcd-login001.rcac.purdue.edu)mit.edu:/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned /ocean/projects/phy240040p/pdong2/light_curve_files

# 3/28

### Accomplished:

- More or less, everything is finished organized
    - Moved everything I dont need into archived folders
    - All the new stuff is new clear folders now
- Still don’t have permanent solution to less GPUs - likely harder to work on ML stuff
- Focusing more on bringing outdated code up to date and making it do all of the new stuff that I wanted it to do
    - Light curve code
    - Feature extraction code
    - TODO: do want to bring continue training file to a nicer place
- Working on writing a journal now - started it but not every detailed yet - only has what I’m currently doing
- Added isolation forest support to transformers, but haven’t run it because it takes a bit of time to get GPUs
    - Currently training another RNN

### Notes:

plot features that are related

- Cornerplot of the features
    - Color code them based on clusters
    - Seaborn pairplot
- Color code it by cluster number
    - See if anyone else has done this before
- Correlation matrix of different feautres
    - Maybe for each cluster

to ssh into new cluster: ssh [pdong2@bridges2.psc.edu](mailto:pdong2@bridges2.psc.edu)

# 3/7

### What I’ve accomplished

- Working on transformer model → tried a few different things, but couldn’t get time to work nicely
    - Just gotta examine dimensions again
- More of training models etc
    - Tried training an RNN with less latent variables (22)
    - Wasn’t able to construct super nicely
- Implemented Isolation forest
    - RNN wasn’t super accurate though, so it was pretty useless
        - Likely need to up latent size → is this true?
- Still PCA’d for like top 6 components and clustered
- Trying to do an overhaul of my code to make it a little bit nicer
- Fixed lightcurve permission thing
    - → haven’t run into it again but that might also just be statistics

### Notes

- Make a project log
- Log what i’ve done so far so i can keep track of what I’m doing
- Need → diary to keep track of features that I tried to do ]

Should check uncertainty → if reconst within uncertainty → tis ok

Isolation forest - just pass in the latest space directly

# 2/27

### Meeting notes:

Some of the fits files don’t seem to have any data in them (HDUL[1]) does not exist

- We should just skip over these
- Try/Ex: - Print out names of files & skip

Change learning rate - sometimes will not work

TSNE - very bad at clustering with distance

PCA - components do mean something

Pass in latent space to go into clustering algorithm

- BUT 40 too large
- Clustering algorithm breaks with too many dimensions

40 things - pass into tree algorithm

HDB SCAN - less than 6

- Reduce latent size of model
- Run PCA - pass in more pca
    - Explained variance - how much each component explains the variance in the data
    - Use 5-6 PCA
- UMAT - try this one as well
- PACMAT - try this one maybe

Implement **batch normalization**:

- During training
- Take the weights of a batch and normalize it
- steps during training to be a little bit faster

Notes - try to reduce the latent size to smaller

Will try to pass in PCA components

- More

Try isolation forest to find outliers

More than 5

Look at likelihood

- Chi Squared
- print the loss functions here

Good plot - histogram of maximum median rate across all sources

histogram of a maximum mean rate across all sources

Histogram of mean count rate / mean average error (prob sym)

- Have the reconstructed light curves have uncertainties on the light curves
- 

Compare Autoencoder to Feature extraction one

Personal notes:

- Need to fix transformer model
- Plot errors
- Normaliize them

- Take some time to just understand the data files

# 2/14

### Accomplished:

Trained a big model - around 16000 lcs

- Am training off of that right now
- Just using MSE
- Show the graphs

Will Train another big model adding in the KL divergence term - unsure how

Read the paper

Mostly testing to try to get the latent space down and tuning hyperparameters

- Will also try testing incorporating different things into the loss function now

### Questions:

- Poissonian Distribution giving me a little bit of trouble
- Do we have the code for the transformer yet?

### Notes:

Change the latent space to be smaller

download and make dummy folder to load data sooner

w&b - online logging tool → watch my training as its going

- make it update every ~100 epochs

# 2/7

What I’ve Done

- Dan and I figured out that it was nothing wrong with the model, just not enough epochs
    - Further testing sees that it stabiilizes around 30,000 so I”ll probably keep it out there for now
- Started training a really big set (1000 light curves)
    - Taking a long time and I think I can simplify the NN to make it faster
- Trying to adapt the poissonian from the paper
- 

Questions

- 
- Can i start working on transformer?
- 

Notes

Read section 3 of paper

- try to just ask them for the code
- 

# 1/31

### Stuff that I’ve done

- Got the error thing to be working
- Implemented the log-likelihood function for a poissonian distribution
- Tried varying tons of factors
- 

### Questions

- Why is the reconstruction literally just the average
    - Posterior Collapse
    - Saying my decoder is ignoring the latent space → probable true
- Latent Space is currently 50 → is that too big if i’m literally only passing in 3 parameters
    - What other things should i try to pass in
    - Am just passing in the numbers straight
- Just been reading up on better ways to handle this
- Adding noise to latent space

### Meeting Notes

FIND OUT WTF IS HAPPENING IN THE ENCODER/DECODER SIDE OF IT

Check out the latent space

Check some bugs

- Encoder / Decoder dimension mismatch
- Output of encoder has Nans → possbile
- check all of the data through each step
- Architecture issue

# 1/23?

### What I’ve accomplished

- Was troubleshooting GPU for a while but then decided to just go to office hours to get it figured out
    - OH was yesterday - got access to a testing partition that has more updated GPUs, meaning that I can run larger datasets now
- Was trying to figure out workflow for future → still a bit weird, but is working right now
    - Will set up jupyter notebook locally later and should be able to run with better gpus → just missing password to nodes
- Also heard about the other cluster, but i think i remember the OH guy saying that its software was somewhat outdated so it was also a hard thing to use
- Adapted the model to take in the lower and upper error, but couldn’t run it yesterday because i was having some issues grabbing the data
    - Seems to be resolved this morning
- Made lots of plots
- Started working on HBDScan clustering algorithm

### Questions

SIMCLAIR Paper - jeffery

# 1/21

1/18

- Got the statistical method to work by ranking the outliers based on if they are the most outlierish
- Restructured the data so that its in a dataframe now

![image.png](Astrophysics%20Meeting%20Notes%20Total/image.png)

TODO:

- Find which features are the most prevalent - not possible with an isolation tree
- Why are skew and kurtosis breaking
- find a better way to rank the outliers ( only doin iso score right now)
- why is the PCA principle component so strong on the axes

### What I did this week:

- All of the data is now in a dataframe - much easier to get information from and analyze individual light curves once the data is done
- Statistical Method now Ranks the outliers based on the isolation forest
    - Will eventually add add stuff from other outlier detection methods as well
- Spent SOO LONG trying to get the GPU to work
    - Got closer → now the only issue is that my pytorch is still out of date with my CUDA so they dont work together
    - Emailed the Engaging help desk so hopefully they can help with that
- Eventually just gave up and ran on CPU - had to limit to only 400 bc anything above that was too slow
    - can see reconstruction error on a specific light curve and as the model continues to train
    - Currently just factoring a 1d vector of rates, but would love to add error into this method
- Outlier Detection
    - Extracted latent space with VAE encoder step
    - Ran this just through an isolation forest and found a few outliers

Questions:

- Had to remove skew and kurtosis - Gave me a Nan Error every single time that I tried to use the pipeline on that
- Why is the first principle component in my PCA graph so much more important than any other feature
- I dont think isolation forest has a ranking of which features were most important → only random forest
- Would really like to start working with slurm
    - Trying to do the reading based on the document riccardo sent, but would love to see maybe a practical demonstration of how it works or maybe some screenshots on how to do it
- 

### Meeting Notes

VAE working

Clipping at 30

Upper limit - Median and lower bound are the same error

(Maximum / minimum)/uncertainty

Just plugg errors ion to the input size - make 2

ONLY CHANGE FOR ENCODERS  NOT DECODERS

Pass in both upper and lower uncertainties, and symmetric

- myabe not the symmetric

Check if they’re percentiles in teh ERRM

Log Llkelihood function to add to MSE for poisoniian

![image.png](Astrophysics%20Meeting%20Notes%20Total/image%201.png)

ERRP - ERRM / 2 - symmetrized error

Median - 16th percentile - lower

rate - rate ERRM - upper uncertainty

rate ERRP - rate - lower uncertainty

Median - 

# 1/17

### Things I’ve Accomplished:

- Troubleshooted the permissions error
- Setup the github
- Found which factors are the most important
    - TODO: rank the plots on what the most outliers are

### Things that I want to ask about:

- Permissions error, can we get that fixed
- How do i use a GPU in jupyter notebook - training the VAE is super slow
    - Check the additional modules

### Notes

- Look at the paper he sent me for some more common features

# 1/14

### Pre-meeting Notes

What I’ve accomplished:

- Started accounting for error in statistics
    - Can show the graphs
- Started the machine learning input
    - Can technically train the auto-encoder on just the rates
    - Q - need help on some of how to visualize this data - not completely understanding the notebook

Questions for them:

- Data still not able to be accessed
- How should I examine “outliers” once I’ve detected them
    - Kinda understand but not
- 

Feature Correlation Matrix

- Iso Forest to see which outlier
    - which factors are most dominant in the outlier
        - (Max/min) / uncertainty (sumed)
- List of features in by Sarah Webb → try to use these

Stuff to do:

- Plot errors
- show which features contribute to the outlier more
    - maybe show on the plot and make some more interesting observations
    - 

# 1/10

### Pre-Meeting Notes

What I’ve accomplished:

- Plotted the light curves - made some functions that made this possible
- Started by analyzing raw data
    - Too hard because it was pretty hard to decide how to make the data all work
        - Ended up just looking at the maximum length and padding the light curve with Rate = 0s after the times are increased past the map
- Analyzed statistics
    - Seemed to be much better - took some random statistics: mean, median, std, max amplitude, beyond 1 sigma, flux percentile…
    - Each one of the light curves became represented by a list of just these statistics
    - Detect outliers
        - Scale them
        - isolation forest and local outlier factor or IQR and std deviation to be classified as an outlier
        - Visualized them on PCA → looked nicer and it seemed like the outliers (red dots) were in the right place

Questions:

- Why do I still not have access to all the files
    - Still getting the same error thrown
- What do you think i should explore next?
    - Should i keep on trying to work with the raw data
    - Kinda want to move onto some more machine learning things → training auto-encoders
    - I have a plot of some outliers → is there some way for me to start cross-referencing them and seeing what they actually are
- Do they know anything about Bayesian Blocks for doing light curves
    - Was talking to a friend who got 2nd in a national research lab by detecting transits from x-ray light curves
    - Procedure?
        - Use bayesian blocks to group the light curves based on when the rates are the same
        - Train a random forest model on the histograms
            - 0 if there are bins that are high low high
            - 1 if there are bins that look kinda close to that
            - (idk what emily said) if the bins aren’t

Meeting Notes:

- Smiley faces → calibration corrections
    - Used fractional exposure to correct for the rate
    - Do i want to remove them with the use of flags or also pass in the variability of the light curve into the mml
- variability
- analyzing
    - difference / variability - over three might be significant
- Notes: Need to account for the variability  the rate  - very important to see if there is actually anything notable
    - Perhaps lots of measures can look like data
- Look at light curves and try to extract meaningful features
    - Maybe only get upper 3rd quartile or something
    - UMAT - instead of PCA
    - Plot → red outlier points light curves
- LSTM Autoencoder - variable length or CNN - static data of fixed length or transformers - variable length
    - Take input light curve and force to generate a light curve
    - Try to build this
    - Maybe use LSTM autoencoder with raw data
        - then also pass in other statistical features to make it any better
        - switch reward function → MSE → likelihood function of a poissonian distribution
            - To account for errors
            - 

# 1/3

**Astrophysics Notes**

Engage - where we can access the data

Data 

- 200000 fits files
- Multiple column
    - Time, intensity (counts per seconds) - most time
    - Time series on roughly 10 data points per source
    - Other dimensions - energy of photon, etc —> secondary stuff
- Either in our own galaxy or farther away

Different methods:

- small number of data points
- Full neural network to try to get the latent space structured with constructive learning
    - Construct latent space → biggest part of this project
    - Do we need a machine learning algo to do this? → maybe not just use statistical
        - perhaps find simpler ways to represent the data
        - Pass into clustering algorithm

This week → explore the data

- fits files → need to learn
    - Store the data and the metadata

Apply clustering algorithms

- After getting the outliers → we cross reference to other catalogues and try to identify what the objects are

General Vibes:

- Riccardo - more specialized in astro
    - Good for expanations that require more astro stuff
- Dan - Very nice, but might think that you’re a bit stupid
    - Very nice for elemenary explanations

FIrst steps:

1. Get set up

engaging

Jupyter or VS code

1. explore data
- plot bunch of things
- see how the data is working
- 

From Emily:

- Need to distinguish transits from flares
    - A ptest of the different light curves in different bands
        - Flares will only show up in low and medium - absent in high
        - if the low med and high are statistically different, there is a flare
- Do **Bayesian Blocks** algorithm for binning the light curves and feeding it in
    - Train a random forest model on the histograms
        - 0 if there are bins that are high low high
        - 1 if there are bins that look kinda close to that
        - (idk what emily said) if the bins aren’t
- After getting all statistically significant x-ray events
    - need to distinguish transits from flares - get all three wavelengths
-