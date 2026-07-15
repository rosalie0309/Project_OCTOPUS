# pipeline_utils.py

import json
from pathlib import Path


# ============================================================
# 1. Formatage lisible des éléments du pipeline
# ============================================================

def format_loss(loss):
    """
    Transforme le nom interne de la fonction de perte en nom lisible.
    """
    mapping = {
        "ce": "CE",
        "ce_dice": "CE+Dice",
        "ce_semeda": "CE+SEMEDA",
    }

    if loss is None:
        return "loss inconnue"

    return mapping.get(loss, str(loss))


def format_preprocess(preprocess):
    """
    Transforme le nom interne du prétraitement en nom lisible.
    """
    if preprocess is None or preprocess == "none":
        return "Aucun prétraitement"

    mapping = {
        "clahe": "CLAHE",
        "expansion_rgb": "Expansion RGB + égalisation",
    }

    return mapping.get(preprocess, str(preprocess))


def infer_section(exp_name):
    """
    Récupère la section expérimentale depuis le début du nom :
    7A, 7B, 8A, 9C, etc.

    Exemple :
    7B_segformerb0_ce_dice -> 7B
    fcn8_1_1_16_run00_fold5 -> -
    """
    first_part = exp_name.split("_")[0]

    if len(first_part) >= 2 and first_part[0].isdigit():
        return first_part

    return "-"


def infer_architecture_from_name(exp_name):
    """
    Déduit l'architecture à partir du nom de l'expérience.
    Utile si le fichier metadata ne contient pas explicitement architecture.
    """
    name = exp_name.lower()

    if "segformerb0" in name:
        return "SegFormer-B0"
    if "segformerb1" in name:
        return "SegFormer-B1"
    if "segformerb2" in name:
        return "SegFormer-B2"

    if "fcn8" in name or "fcn8vgg16" in name:
        return "VGG16-FCN8"

    if "unet" in name:
        return "U-Net"

    return "Architecture inconnue"


def infer_variant_from_name(exp_name):
    """
    Déduit la variante SegFormer à partir du nom du fichier.
    """
    name = exp_name.lower()

    if "segformerb0" in name:
        return "b0"
    if "segformerb1" in name:
        return "b1"
    if "segformerb2" in name:
        return "b2"

    return None


def checkpoint_to_exp_name(ckpt_path):
    """
    Convertit un fichier checkpoint en nom d'expérience.

    Exemples :
    7B_segformerb0_ce_dice_best.pth -> 7B_segformerb0_ce_dice
    fcn8_1_1_16_run00_fold5.pth -> fcn8_1_1_16_run00_fold5
    """
    name = ckpt_path.name

    if name.endswith("_best.pth"):
        return name.replace("_best.pth", "")

    if name.endswith(".pth"):
        return name.replace(".pth", "")

    return ckpt_path.stem


def metadata_to_exp_name(meta_path):
    """
    Convertit un fichier metadata en nom d'expérience.

    Exemple :
    7B_segformerb0_ce_dice_metadata.json -> 7B_segformerb0_ce_dice
    """
    return meta_path.name.replace("_metadata.json", "")


def make_pipeline_label(pipeline):
    """
    Crée le label lisible affiché dans le navigateur.
    """

    exp_name = pipeline["experiment_name"]

    section = infer_section(exp_name)
    architecture = pipeline.get("architecture", "Architecture inconnue")
    loss = format_loss(pipeline.get("loss"))
    preprocess = format_preprocess(pipeline.get("preprocess"))

    img_h = pipeline.get("img_h")
    img_w = pipeline.get("img_w")

    if img_h is not None and img_w is not None:
        resolution = f"{img_h}×{img_w}"
    else:
        resolution = "résolution inconnue"

    fold = pipeline.get("fold")

    # Cas validation croisée
    if fold is not None:
        return (
            f"{architecture} | Fold {fold} | {loss} | "
            f"{resolution} | {preprocess}"
        )

    # Cas expériences classiques : 7A, 7B, 9C...
    if section != "-":
        return (
            f"{architecture} | {loss} | {resolution} | "
            f"{preprocess} | {section}"
        )

    # Cas fallback : VGG16-FCN8 ou autre
    return (
        f"{architecture} | {loss} | {resolution} | "
        f"{preprocess}"
    )


# ============================================================
# 2. Découverte automatique des pipelines
# ============================================================

def discover_pipelines(models_dir="models"):
    """
    Détecte automatiquement les pipelines disponibles à partir du dossier models/.

    Le dossier doit contenir les paires :
    - nom_experience_best.pth
    - nom_experience_metadata.json

    ou, pour certains modèles comme FCN8 :
    - nom_experience.pth
    - nom_experience_metadata.json

    La fonction retourne un dictionnaire :
    {
        "7B_segformerb0_ce_dice": {
            "experiment_name": ...,
            "checkpoint_path": ...,
            "metadata_path": ...,
            "loss": ...,
            "preprocess": ...,
            ...
        }
    }
    """

    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(f"Le dossier models est introuvable : {models_dir}")

    # Tous les checkpoints .pth
    checkpoint_files = list(models_dir.glob("*.pth"))

    # Tous les fichiers metadata
    metadata_files = list(models_dir.glob("*_metadata.json"))

    # Index des checkpoints par nom d'expérience
    checkpoints = {}
    for ckpt_path in checkpoint_files:
        exp_name = checkpoint_to_exp_name(ckpt_path)
        checkpoints[exp_name] = ckpt_path

    # Index des metadata par nom d'expérience
    metadata = {}
    for meta_path in metadata_files:
        exp_name = metadata_to_exp_name(meta_path)
        metadata[exp_name] = meta_path

    # On garde seulement les expériences qui ont checkpoint + metadata
    common_exp_names = sorted(set(checkpoints.keys()) & set(metadata.keys()))

    pipelines = {}

    for exp_name in common_exp_names:
        ckpt_path = checkpoints[exp_name]
        meta_path = metadata[exp_name]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        config = meta.get("config", {})

        # Architecture
        architecture = config.get("architecture")
        if architecture is None:
            architecture = infer_architecture_from_name(exp_name)

        # Variante SegFormer
        variant = config.get("variant")
        if variant is None:
            variant = infer_variant_from_name(exp_name)

        # Résolution
        img_h = config.get("img_h")
        img_w = config.get("img_w")

        # Fonction de perte
        loss = config.get("loss")

        # Prétraitement
        preprocess = config.get("preprocess")

        fold = infer_fold_from_name(exp_name, meta)

        # Construction du pipeline
        pipeline = {
            "experiment_name": exp_name,

            "checkpoint_path": str(ckpt_path),
            "metadata_path": str(meta_path),

            "architecture": architecture,
            "variant": variant,

            "loss": loss,
            "img_h": img_h,
            "img_w": img_w,
            "preprocess": preprocess,

            "batch_size": config.get("batch_size"),
            "lr": config.get("lr"),
            "weight_decay": config.get("weight_decay"),

            # Seuils sauvegardés
            "val_threshold": meta.get("val_threshold"),
            "test_threshold": meta.get("test_threshold"),

            # Métriques sauvegardées dans le metadata
            "val_metrics": meta.get("val_metrics", {}),
            "test_metrics": meta.get("test_metrics", {}),

            # Configuration complète au cas où
            "config": config,
            # gestion des folds pour les pipelines k-fold
            "fold": fold,
            "is_kfold": fold is not None,
        }

        pipeline["section"] = infer_section(exp_name)
        pipeline["label"] = make_pipeline_label(pipeline)
        pipeline["kfold_group_key"] = make_kfold_group_key(pipeline)

        pipelines[exp_name] = pipeline

    return pipelines


# ============================================================
# 3. Fonctions utilitaires d'affichage des métriques
# ============================================================

def format_metric_value(value):
    """
    Formate les métriques pour affichage.
    Si la valeur est entre 0 et 1, on l'affiche en pourcentage.
    """
    if value is None:
        return "-"

    if isinstance(value, (int, float)):
        return f"{value * 100:.2f} %"

    return str(value)


def get_main_metrics(metrics):
    """
    Extrait seulement les métriques principales à afficher dans l'application.
    """
    if metrics is None:
        return {}

    wanted_keys = [
        "accuracy",
        "precision",
        "recall",
        "fallout",
        "recall_minus_fallout",
        "f1",
        "dice",
        "iou",
        "iou_background",
        "miou",
    ]

    return {
        key: metrics[key]
        for key in wanted_keys
        if key in metrics
    }

def infer_fold_from_name(exp_name, meta=None):
    """
    Récupère le numéro de fold depuis le metadata ou depuis le nom du fichier.
    """

    if meta is not None and "fold" in meta:
        return meta["fold"]

    name = exp_name.lower()

    if "fold" in name:
        after = name.split("fold", 1)[1]
        digits = ""

        for char in after:
            if char.isdigit():
                digits += char
            else:
                break

        if digits:
            return int(digits)

    return None

def make_kfold_group_key(pipeline):
    """
    Clé permettant de vérifier que plusieurs folds appartiennent
    bien au même protocole expérimental.
    """

    return (
        pipeline.get("architecture"),
        pipeline.get("variant"),
        pipeline.get("loss"),
        pipeline.get("img_h"),
        pipeline.get("img_w"),
        pipeline.get("preprocess"),
    )

# ============================================================
# 4. Test rapide du module
# ============================================================

if __name__ == "__main__":
    pipelines = discover_pipelines("models")

    print(f"{len(pipelines)} pipeline(s) détecté(s).\n")

    for exp_name, pipeline in pipelines.items():
        print(f"- {exp_name}")
        print(f"  Label       : {pipeline['label']}")
        print(f"  Checkpoint  : {pipeline['checkpoint_path']}")
        print(f"  Metadata    : {pipeline['metadata_path']}")
        print(f"  Val threshold : {pipeline['val_threshold']}")
        print()