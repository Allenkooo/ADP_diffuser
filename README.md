# ADP README

## Data

- Dataset for AntiDMPpred

```linux=
~ko123chungg/git/ADP_repo/Data/ADMP/positive_negative.fasta
```

- Dataset for ADP-Fuse
```linux=
~ko123chungg/git/ADP_repo/Data/ADP_fuse/
```

## Trained Model

- Word2Vec

```linux=
~ko123chungg/git/ADP_repo/Script/outputs/Word2Vec_model/9_123_10_20_1.pt
```

- Diffusion model

```linux=
~ko123chungg/git/ADP_repo/Script/outputs/Diffusion_model/w2v_9mer_pred_x0_pad_100000/model-3.pt
```

## Main Script

### Device: Merry04

- Main scrpit file
```linux=
~ko123chungg/git/ADP_repo/Script/main.py
```

- Config file
    - Setting of Wor2Vec, diffusion model, and other training or inference details
    - Details of the config file you can check the create_parser() in main.py
    ```linux=
    ~ko123chungg/git/ADP_repo/Script/args.yaml
    ```

    Below is a example of config file
    ```yaml=
    Data_path: /home/ko123chungg/git/ADP_repo/Data/ADMP/positive_negative.fasta
    Word2Vec_args:
     model_path: /home/ko123chungg/git/ADP_repo/Script/outputs/Word2Vec_model/
     emb_dim: 128
     kmer: 9
     window_size: 20
     epochs: 10
     sg: 1           # Skipgram:1, CBow: 0
    Unet_args:
     dim: 64
     dim_mults: [1, 2, 4]
     channels: 128   # output dim
    Diffusion_args:
     seq_length: 32        # L - kmer + 1
     timesteps: 1000
     sampling_timesteps: 999
     pe: False             # positional encoding
     objective: pred_x0
    results_folder: /home/ko123chungg/git/ADP_repo/Script/outputs/Diffusion_model/w2v_9mer_pred_x0_pad_100000/
    checkpoint: 3      # checkpoint diffusion model to load
    n_samples: 1000    # number of generated data
    labeling: KNN      # KNN, IC(Interal Classifer), TD(Two Diffusers)
    Select_method: Euclidean   # Euclidean distance, kmeans
    ```
- To run main file:

    - Train Word2Vec model
    ```linux=
    python main.py -c args.yaml -tw
    ```
    - Train Word2Vec model
    ```linux=
    python main.py -c args.yaml -td
    ```
    - Train Random forest
    ```linux=
    python main.py -c args.yaml -tr
    ```
    - Train Random forest with augmented data
    ```linux=
    python main.py -c args.yaml -tr -aug
    ```
