# MP3 Spectral Refiner

Outil en ligne de commande pour **affiner le spectre** de fichiers MP3 compressés : EQ correctif adaptatif, dynamique légère, et préservation du niveau perçu (pas de sur-boost).

Inspiré du type d'outil proposé par [Soundplate MP3 Enhancer](https://soundplate.com/mp3-enhancer/), mais orienté **cohérence fréquentielle** plutôt que mastering loud.

## Prérequis

- Python 3.10+
- FFmpeg (installé dans le `PATH`, ou fourni automatiquement via `imageio-ffmpeg`)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Fichier unique

```bash
# Affinage spectral (défaut : preset gentle, loudness préservée)
python enhance.py "1-01 Hidden Hills.mp3"

# Sortie personnalisée
python enhance.py input.mp3 -o output.enhanced.mp3

# Presets : gentle (défaut), balanced, aggressive
python enhance.py input.mp3 --preset balanced

# Analyse seule (sans export)
python enhance.py input.mp3 --dry-run

# Mode loud legacy (makeup + normalisation peak)
python enhance.py input.mp3 --boost

# IA optionnelle (artefacts + super-res)
python enhance.py input.mp3 --ai

# Verbose
python enhance.py input.mp3 -v
```

### IA (optionnel)

Installation des dépendances IA (~2–4 Go de modèles au premier lancement) :

```bash
pip install -r requirements-ai.txt
```

**Python 3.12+ (Windows)** : DeepFilterNet et AudioSR n'ont pas de wheels fiables. Le pipeline utilise automatiquement des **fallbacks** adoucis :
- Phase 1 artefacts → `noisereduce` (spectral, wet défaut 0.5)
- Phase 2 super-res → EQ adaptatif hautes fréquences (gains plafonnés)

Pour les modèles complets, utilisez **Python 3.11** :

```bash
pip install -r requirements-ai-deepfilter.txt   # + DeepFilterNet3
pip install -r requirements-ai-superres.txt     # + AudioSR
```

| Commande | Effet |
|----------|-------|
| `python enhance.py track.mp3 --ai` | Mode auto : artefacts si MP3 compressé, super-res si roll-off sévère + GPU |
| `python enhance.py track.mp3 --ai --ai-mode artifacts` | Réduction d'artefacts seulement |
| `python enhance.py track.mp3 --ai --ai-mode full` | Artefacts + AudioSR |
| `python enhance.py --batch --ai` | Batch avec modèles chargés une seule fois |
| `python enhance.py --batch --ai --ai-fast` | Batch rapide : skip super-res sur CPU |

Options IA :

- `--ai-mode` : `auto`, `artifacts`, `superres`, `full`
- `--ai-device` : `auto`, `cpu`, `cuda`
- `--ai-wet` : mix wet/dry phase artefacts (défaut `0.5`)
- `--ai-fast` : skip AudioSR sur CPU
- `--ai-force-superres` : forcer AudioSR même sur CPU

### Traitement par lot

Placez vos MP3 dans le dossier `inputs/` à la racine du projet, puis :

```bash
python enhance.py --batch
# ou simplement (sans argument)
python enhance.py
```

Les fichiers affinés sont écrits dans `outputs/` avec le suffixe `.enhanced.mp3`.
Les tags ID3 (artiste, album, etc.) sont conservés depuis chaque fichier source.

## Philosophie de traitement

- **Corriger** plutôt qu'amplifier : coupes prioritaires, boosts plafonnés
- **Préserver le niveau** : RMS sortie ≈ RMS source (+1 dB max si piste très faible)
- **Limiter** uniquement en sécurité (soft ceiling), pas comme outil de loudness
- **`--boost`** : retrouve l'ancien comportement plus loud si besoin

## Tests

```bash
python -m pytest tests/ -v
```

Placez `1-01 Hidden Hills.mp3` à la racine du projet pour les tests manuels et automatisés.
