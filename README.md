# artist2lyrics

Generate lyrics in the style of your favorite artists  :notes:

## Introduction

**artist2lyrics** is a set of CLI tools that lets you create your own lyrics generator AI, simply by specifiying the artists whose lyrics style you want your AI to imitate. Lyrics crawling, pre-processing, word embeddings generation, model training, and lyrics generation - all basic steps of the lyrics generation pipeline are already handled by this tool! By default, artist2lyrics uses a 2-layer stacked LSTM RNN with dropout to learn lyrics in a word-based manner.

Well then, let's see how this works.

### OPM (Original Philippine Music) Lyrics

To start off, I'll be using music from my country. Using artist2lyrics, I trained a model on Tagalog songs of three of my loved Filipino bands: SugarFree, Itchyworms, and Eraserheads. Now, generating some lyrics...
```
barkada . hawak ang 'yong kamay habang ay hindi . naaalala ka pa .  
ilaw ikaw sarili ng mundo ng ilalim ng buwan ilalim ng buwan .
```
Hmmm... so this AI seems to be thinking about his friends, a loved one, and the moon.  
Let's generate some more.
```
bituin sa ating pugad , sa ating pugad sa ating pugad , sa ating pugad .  
bawat sandali ng aking buhay puno mo ang aking kamay .
```
OK that's pretty sweet. This AI's feeling at home with a loved one.
```
. kahapon lang tayo ay buhay ay yakapin , ako ay buhay buhay ako  
hay buhay nga naman hay buhay nga naman .
```
Now it's frustrated about life.  
Let's try something else.

### Air Supply Lyrics

After training a model on more than 150 songs of Air Supply, we get these:
```
see stand it the feeling when you're far away  
i want to take you up in my little world so you'll be here today .  
when we fly through the night can't you feel the words is the truth of the line .
```
```
let what being here with you taking what is mine just between the tears .  
i could not more cry i say that i've i thought in heart alright  
but that heart is a face , can't you thought know just tonight i wanna survive you
```
Well this AI's kinda weird and funny, but surely it wants to sing about love.  
Let's try a different genre.

### MCR (My Chemical Romance) Lyrics

```
right after me you want to kick up you're dead ? .  
so play a last to remember to the head . i will want .  
so maybe a sad song with nothin' to do  
about a life long wait for a hospital about a life  
about well better know get a our way it's .
```
```
good is you , you , you . me and you and all of these living dead ,  
burning up in the sun where the bodies go , i've on , and on ,  
and on , and on , and on . so the red your - that what's inside  
you'll a worst from have out gone .
```
Great. I have just raised an emo kid AI. :laughing:  

Okay, generating lyrics sure is fun. Now let's get to more FUN!

## Getting Started

### Prerequisites

* PostgreSQL
* Python3
    * numpy, tensorflow, keras, gensim, matplotlib, psycopg2, scikit_learn
    * see [requirements.txt](requirements.txt) for specific versions used

### Database

By default, artist2lyrics uses _postgres_ database to store the crawled lyrics.  
To be able to use this database, you need to register your user account as a user:
```
sudo -u postgres createuser <user>
```

## Usage

The pipeline for this tool's lyrics generation is as follows:  
_Lyrics Crawling_ -> _Pre-processing_ -> _Embeddings Generation_ -> _Model Training_ -> _Lyrics Generation_

Each of these are handled by a particular Python script. Before running these, make sure to set desired configuration in _artist2lyrics.cfg_. Importantly, to specify which artists you'd want to get lyrics from, set the "artists" field in the config file  
in comma separated format. For example:
```
artists = airsupply,mychemicalromance
```
Note that for lyrics crawling, we are gathering data from [azlyrics](https://www.azlyrics.com/). The specified names of the artists shall match the name appearing in azlyrics url. For example:
```
https://www.azlyrics.com/m/mychemicalromance.html
https://www.azlyrics.com/a/airsupply.html
```
With these set, you may proceed with execution of the scripts in the following sequence:
1. **crawl_lyrics.py** - lyrics crawling
2. **process_data.py** - data pre-processing and embeddings generation
3. **train_model.py** - model training
4. **generate_lyrics.py** - use trained model for lyrics generation

## Built With

* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [gensim](https://radimrehurek.com/gensim/)

## Authors

* **Christopher John Bayron**
    * [Github](https://github.com/cjbayron)
    * [LinkedIn](https://www.linkedin.com/in/christopher-john-bayron)

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

Credits to 
* [IIITV org](https://github.com/iiitv) for the [lyrics-crawler](https://github.com/iiitv/lyrics-crawler)
* [Kyubyong](https://github.com/Kyubyong) for [word2vec demo](https://github.com/Kyubyong/wordvectors)
* [Jeff Delaney](https://www.kaggle.com/jeffd23) for [t-SNE demo](https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne/notebook)