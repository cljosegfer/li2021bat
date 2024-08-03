# reproducao code

primeiro edite o <train_beat_aligned_swin_transformer.json> como desejar:
* n_gpu (linha 3)
* batch_size (linha 37)
* epochs (linha 70)

dps rode o <create_codeh5.py> mudando os caminhos como desejar:
* code15 (linha 21, caminho do h5)
* metadata (linha 22, caminho do exams.csv do code15!!!!!!!!!)

dps rode o <main_beat_aligned.py>:
> python main_beat_aligned.py -c train_beat_aligned_swin_transformer.json -d 0 -s 1

o resultado vai estar num json em <output/saved/results/beat_aligned_swin_transformer_CODE/{ultimo folder}/metrics.json>

teoricamente é pra dar certo :)

# Beat-aligned Transformer

This reposity contains the code for the paper "BaT: Beat-aligned Transformer for Electrocardiogram Classification" on the [Physionet/CinC Challenge 2020 dataset](https://physionetchallenges.org/2020).

The dependencies are listed in the requirements.txt. 

To run this code, you need to download and organize the challenge data, and set corresponding paths in the config json file. Then, utilize preprare_segments.py to generate heartbeat segments and resample ratios, and run:
```
python main_beat_aligned.py -c train_beat_aligned_swin_transformer.json -d 0 -s 1
python main_beat_aligned.py -c train_swin_transformer.json -d 1 -s 1
``` 
c for the config json file path, d for the gpu device index, and s for the random seed.

