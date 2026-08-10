"""六類任務的標籤對映與守門。

新增 `--task multiclass` 之後最容易出的錯不是訓練壞掉，是**標籤對錯**：
六類的索引若與 CLASSES 順序不一致，模型照樣訓練、照樣給出好看的 macro F1，
只是每個類別名字都掛錯人。這種錯不會有錯誤訊息，只能靠測試釘住。

另一件要釘的是**預設值**：README 與 results/ 的數字全部產於 binary，
預設一旦被改成 multiclass，既有結論就對不上了。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tiles import CLASSES  # noqa: E402
from train import target_of  # noqa: E402


def item(label: str) -> dict:
    return {"path": Path(f"/tmp/{label}.jpg"), "label": label, "exp": "exp1",
            "num": 1, "is_defect": int(label != "Free")}


def test_classes_are_the_datasets_six() -> None:
    assert CLASSES == ["Blowhole", "Break", "Crack", "Fray", "Uneven", "Free"]


@pytest.mark.parametrize("label", CLASSES)
def test_multiclass_target_is_the_class_index(label: str) -> None:
    assert target_of(item(label), "multiclass") == CLASSES.index(label)


@pytest.mark.parametrize("label", CLASSES)
def test_binary_target_is_defect_flag(label: str) -> None:
    assert target_of(item(label), "binary") == (0 if label == "Free" else 1)


def test_the_two_tasks_disagree() -> None:
    """探針：若 target_of 忽略 task 參數，上面兩組測試可能同時假綠。"""
    diff = [c for c in CLASSES
            if target_of(item(c), "binary") != target_of(item(c), "multiclass")]
    assert len(diff) >= 4, "兩種任務的標籤幾乎一致，target_of 可能沒真的分流"


def test_default_task_is_binary() -> None:
    """README 與 results/ 的數字全部產於 binary——預設值換掉，那些數字就對不上。"""
    src = (Path(__file__).resolve().parent.parent / "src" / "train.py").read_text(encoding="utf-8")
    assert 'choices=["binary", "multiclass"], default="binary"' in src, (
        "--task 的預設值不再是 binary。若這是有意的改動，請同步重跑 README 的數字，"
        "並更新本測試；否則既有結論會與程式碼脫節。")


def test_multiclass_writes_to_a_separate_results_file() -> None:
    """六類結果若落進 training_results.json，會被 analyze_results.py 與二分類混算。"""
    src = (Path(__file__).resolve().parent.parent / "src" / "train.py").read_text(encoding="utf-8")
    assert 'a.out = "training_results_multiclass.json"' in src
