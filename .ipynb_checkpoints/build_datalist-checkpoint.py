import os


def build_train_list(root, patients):
    """Single-modality (FLAIR only) train list."""
    data = []
    for p in patients:
        p_dir = os.path.join(root, "train", p)
        for t in os.listdir(p_dir):
            t_dir = os.path.join(p_dir, t)
            data.append({
                "image":     os.path.join(t_dir, f"{p}_{t}_FLAIR.nii.gz"),
                "label":     os.path.join(t_dir, f"{p}_{t}_MASK.nii.gz"),
                "patient":   p,
                "timepoint": t,
            })
    return data


def build_train_list_mm(root, patients):
    """Multi-modal (FLAIR + T1 + T2) train list."""
    data = []
    for p in patients:
        p_dir = os.path.join(root, "train", p)
        for t in os.listdir(p_dir):
            t_dir = os.path.join(p_dir, t)
            data.append({
                "flair":     os.path.join(t_dir, f"{p}_{t}_FLAIR.nii.gz"),
                "t1":        os.path.join(t_dir, f"{p}_{t}_T1.nii.gz"),
                "t2":        os.path.join(t_dir, f"{p}_{t}_T2.nii.gz"),
                "label":     os.path.join(t_dir, f"{p}_{t}_MASK.nii.gz"),
                "patient":   p,
                "timepoint": t,
            })
    return data


def build_test_list(root):
    """Single-modality (FLAIR only) test list."""
    data = []
    test_root = os.path.join(root, "test")
    for p in sorted(os.listdir(test_root)):
        p_dir = os.path.join(test_root, p)
        data.append({
            "image":   os.path.join(p_dir, f"{p}_FLAIR.nii.gz"),
            "label":   os.path.join(p_dir, f"{p}_MASK.nii.gz"),
            "patient": p,
        })
    return data


def build_test_list_mm(root):
    """Multi-modal (FLAIR + T1 + T2) test list."""
    data = []
    test_root = os.path.join(root, "test")
    for p in sorted(os.listdir(test_root)):
        p_dir = os.path.join(test_root, p)
        data.append({
            "flair":   os.path.join(p_dir, f"{p}_FLAIR.nii.gz"),
            "t1":      os.path.join(p_dir, f"{p}_T1.nii.gz"),
            "t2":      os.path.join(p_dir, f"{p}_T2.nii.gz"),
            "label":   os.path.join(p_dir, f"{p}_MASK.nii.gz"),
            "patient": p,
        })
    return data
