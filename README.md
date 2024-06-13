# AutoML for land cover classification using satellite earth observation products: application to environmental environments in Andalusia.


## Install dependencies

```sh
cd lancoverpy
pip install -e .
```

## To run evolutive algorithm run this command:

Within this file you can modify the size of the population or the number of evaluations, you can also modify the name of the folder with the output files.

```sh
python /src/landcoverpy/run_singleobjetive_evolutive_algorithm.py
```

 ## To train the base/calculated nueronal network and/or the network obtained from doing RF

Execute the function you need. You can also generate a tile classification using models obtained by RF and the neural network. 

 ```sh
python /src/tfm/main.py
```


## License
This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for more info.
