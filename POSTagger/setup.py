from setuptools import setup, find_packages

setup(
    name='sanskrit_pos_tagger',                      
    version='0.1.0',                                  
    author='Adipta Gain',                             
    author_email='adipta.gain@gmail.com',            
    description='A modular, exportable Python package for Sanskrit Part of Speech (POS) tagging using RNN based architectures (SimpleRNN, LSTM, GRU, BiLSTM) with FastText embeddings and the JNU tagset as base data. This library provides training, inference, and evaluation utilities and user can modify base data, model architecture, etc. to his/her liking. It can give POS tagged result, validate against given tagged result, and also build a model that you can fine-tune as per your requirements.',    
    url='https://github.com/Adigain/SanskritPOSTagger.git',  
    packages=find_packages(exclude=['tests', 'docs']),
    include_package_data=True,                        
    install_requires=[                                
        'tensorflow=2.19.0',
        'numpy=1.26.4',
        'pandas=2.2.2',
        'matplotlib=3.9.2',
        'fasttext',
        'sklearn=1.5.1'
    ],
    classifiers=[                                    
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    keywords='sanskrit nlp pos-tagging bi-lstm fasttext',
)
