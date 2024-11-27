# Object Removal by Exemplar-Based Inpainting

## Utilisation
Le programme principal est accessible via le fichier main.py. Voici la syntaxe générale :

```bash
python3 main.py <path_image> <patch_size> <options>
```
### Arguments :
#### - Obligatoires
```<path_image>``` : Chemin vers l'image à traiter.
```<patch_size>``` : Taille des patchs utilisée pour le traitement (entier).

#### - Optionnels
```--mode``` : Mode de recherche utilisé. Peut être : Local (par défaut) ou Full.
```--dsf``` : Facteur de sous-échantillonnage (entier).
```--mask``` : Chemin vers un fichier .npy contenant un masque à utiliser.


### Exemple
```bash
python3 main.py image.png 16 --mode Full --dsf 2 --mask masque.py
```

