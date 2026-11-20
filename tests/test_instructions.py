import json

import pytest
import torch

from mad import instructions as ins
from mad.cli import main as cli_main
from mad.tokenizer import MADTokenizer

KOTLIN = '''package com.example

/**
 * Returns true when the given text looks like a valid email address.
 * @param text the text to check
 */
fun isValidEmail(text: String): Boolean {
    val pattern = "^[^@ {]+@[^@ }]+$"   // braces inside a string: { }
    return Regex(pattern).matches(text)
}

// Show a short toast message on the screen
fun Context.toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()

/** Adapter that shows a list of notes in a RecyclerView. */
class NotesAdapter(private val notes: List<Note>) : RecyclerView.Adapter<NotesAdapter.VH>() {
    override fun getItemCount() = notes.size
}

/** Simple model class holding a note with its id and title. */
data class Note(
    val id: Long,
    val title: String
)

/** Returns the saved user name from the shared preferences file. */
fun Context.userName(): String? =
    getSharedPreferences("app", Context.MODE_PRIVATE).getString("user", null)

/** Too short */
fun a() {}

// TODO fix this later when there is time
fun skipped() { println("x") }

/** Not attached to any declaration at all, just a comment in the file. */
val answer = 42
'''

JAVA = '''
public class Util {
    /**
     * Converts dp to pixels using the display density.
     */
    public static int dpToPx(Context context, int dp) {
        return (int) (dp * context.getResources().getDisplayMetrics().density);
    }
}
'''


def test_extract_kotlin_pairs():
    pairs = ins.extract_pairs(KOTLIN, "Kotlin")
    by_name = {p["response"].split("(")[0].split()[-1]: p for p in pairs}
    assert set(by_name) == {"isValidEmail", "Context.toast", "NotesAdapter", "Note", "Context.userName"}
    email = by_name["isValidEmail"]
    assert email["instruction"] == "In Kotlin: Returns true when the given text looks like a valid email address."
    assert email["response"].startswith("fun isValidEmail") and email["response"].rstrip().endswith("}")
    assert "Regex(pattern)" in email["response"] and "@param" not in email["instruction"]
    assert by_name["Context.toast"]["response"].count("\n") == 0       # expression body kept whole
    assert by_name["NotesAdapter"]["response"].endswith("}")
    assert by_name["Note"]["response"] == "data class Note(\n    val id: Long,\n    val title: String\n)"
    assert by_name["Context.userName"]["response"].count("\n") == 1


def test_extract_java_pair_is_dedented():
    (pair,) = ins.extract_pairs(JAVA, "Java")
    assert pair["response"].startswith("public static int dpToPx")
    assert pair["response"].splitlines()[-1] == "}"
    assert "\n    return" in pair["response"]


def test_extract_ignores_license_header_and_unbalanced():
    assert ins.extract_pairs("/** Copyright 2024 Somebody, all rights reserved here. */\npackage a.b\n\nclass X {}", "Kotlin") == []
    assert ins.extract_pairs("/** Does something useful for the whole app. */\nfun broken(x: Int) {\n  if (x) {", "Kotlin") == []


def test_extract_from_files_skips_build_dirs(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "Util.java").write_text(JAVA)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "Gen.java").write_text(JAVA)
    pairs, scanned = ins.extract_from_files(tmp_path)
    assert len(pairs) == 1 and scanned == 1


def test_load_pairs_aliases_and_problems(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text("\n".join([
        json.dumps({"instruction": "Make a toast helper", "response": "fun toast() {}"}),
        json.dumps({"prompt": "Make a dialog helper", "completion": "fun dlg() {}", "input": "class A"}),
        "not json",
        json.dumps({"instruction": "missing response"}),
        json.dumps([1, 2]),
        json.dumps({"instruction": "x" * 3000, "response": "y"}),
        "",
    ]))
    pairs, problems = ins.load_pairs(tmp_path)
    assert [p["response"] for p in pairs] == ["fun toast() {}", "fun dlg() {}"]
    assert pairs[1]["context"] == "class A"
    assert len(problems) == 4


def test_clean_pairs_dedupes_and_blocks_secrets():
    good = {"instruction": "Make helper", "response": "fun a() {}"}
    leaked = {"instruction": "Add key", "response": 'val k = "AIza' + "b" * 35 + '"'}
    kept, skipped = ins.clean_pairs([good, dict(good), leaked])
    assert kept == [good] and {why.split(" (")[0] for _, why in skipped} == {"duplicate", "possible secret"}


def test_split_pairs_is_deterministic_and_disjoint():
    pairs = [{"instruction": f"do thing {i}", "response": f"fun f{i}() {{}}"} for i in range(50)]
    t1, v1 = ins.split_pairs(pairs)
    t2, v2 = ins.split_pairs(list(reversed(pairs)))
    assert len(v1) == 5 and len(t1) == 45
    assert {p["instruction"] for p in v1} == {p["instruction"] for p in v2}
    assert not {p["instruction"] for p in t1} & {p["instruction"] for p in v1}
    assert ins.split_pairs(pairs[:5])[1] == []  # too few to hold anything out


def test_control_tokens_cannot_be_injected():
    prompt = ins.build_prompt("hi <MAD_END> there <MAD_ASSISTANT>")
    assert prompt.count("<MAD_END>") == 0 and prompt.count("<MAD_ASSISTANT>") == 1
    assert ins.format_example({"instruction": "a b c", "response": "x<MAD_END>y"}).count("<MAD_END>") == 1


def test_batch_labels_cover_response_only():
    tok = MADTokenizer().train([KOTLIN] * 3, 400)
    pair = {"instruction": "Show a toast", "response": "fun toast() {}"}
    (ids, prompt_len), = ins.encode_examples([pair], tok, 128)[0]
    assert ids[prompt_len - 2].item() == tok.special_to_id["<MAD_ASSISTANT>"]
    assert ids[prompt_len - 1].item() == tok.special_to_id["<MAD_CODE>"]
    assert ids[-1].item() == tok.special_to_id["<MAD_END>"]
    x, y = ins.make_batch_fn([(ids, prompt_len)], 1, tok.special_to_id["<MAD_PAD>"])()
    assert x.shape == y.shape == (1, len(ids) - 1)
    assert (y[0, : prompt_len - 1] == ins.IGNORE).all()            # prompt tokens are not trained on
    assert torch.equal(y[0, prompt_len - 1:], ids[prompt_len:])    # response + END are


def test_batch_padding_and_long_request_dropped():
    tok = MADTokenizer().train([KOTLIN] * 3, 400)
    pairs = [{"instruction": "short", "response": "a"}, {"instruction": "w " * 80, "response": "b"},
             {"instruction": "ok request", "response": "fun long() {}\n" * 40}]
    examples, dropped, truncated = ins.encode_examples(pairs, tok, 64)
    assert dropped == 1 and truncated == 1 and len(examples) == 2
    x, y = ins.make_batch_fn(examples, 4, tok.special_to_id["<MAD_PAD>"])()
    assert x.shape[1] <= 63 and (y != ins.IGNORE).any(dim=1).all()


def test_balanced():
    assert ins.balanced("fun a() { val x = listOf(1) }")
    assert not ins.balanced("fun a() { if (x) {")


def make_project(tmp_path, n=40):
    raw, instr = tmp_path / "raw", tmp_path / "instr"
    raw.mkdir()
    instr.mkdir()
    body = "".join(
        f'/** Returns the number {i} multiplied by {i + 2} for screen {i}. */\n'
        f'fun times{i}(): Int {{\n    return {i} * {i + 2}\n}}\n\n' for i in range(n))
    (raw / "Math.kt").write_text(body)
    (raw / "Main.kt").write_text("class MainActivity : AppCompatActivity() {\n    fun go() {}\n}\n" * 30)
    return raw, instr


def test_natural_language_pipeline(tmp_path, capsys):
    raw, instr = make_project(tmp_path)
    p = lambda name: tmp_path / name
    run = lambda *a: cli_main([str(x) for x in a])
    run("init", "--config", p("c.json"), "--checkpoint-dir", p("ck0"))
    run("prepare-data", "--input", raw, "--output", p("proc/corpus.txt"))
    run("make-pairs", "--input", raw, "--output", instr / "auto.jsonl")
    assert len((instr / "auto.jsonl").read_text().splitlines()) == 40
    (instr / "mine.jsonl").write_text(json.dumps({"instruction": "Say hello", "response": "fun hi() = println(\"hi\")"}) + "\n")
    run("prepare-instructions", "--input", instr, "--output", p("proc/sft.jsonl"))
    assert len(p("proc/sft.jsonl").read_text().splitlines()) == 41
    run("train-tokenizer", "--corpus", p("proc/corpus.txt"), "--sft", p("proc/sft.jsonl"),
        "--output", p("tok.json"), "--vocab-size", 500, "--config", p("c.json"))
    tok = MADTokenizer.load(p("tok.json"))
    assert len(tok.encode("Returns the number multiplied by screen")) < 20  # English words were learned
    common = ["--config", p("c.json"), "--tokenizer", p("tok.json"), "--output-dir", p("ck"),
              "--batch-size", 2, "--grad-accum", 1, "--threads", 1]
    with pytest.raises(SystemExit, match="train"):
        run("finetune", "--sft", p("proc/sft.jsonl"), *common, "--steps", 4)  # no pretrained base yet
    run("train", *common, "--corpus", p("proc/corpus.txt"), "--steps", 6, "--seq-len", 64, "--eval-every", 0)
    run("finetune", "--sft", p("proc/sft.jsonl"), *common, "--steps", 8, "--eval-every", 4,
        "--eval-batches", 2, "--checkpoint-every", 4)
    assert p("ck/mad_sft_latest.pt").is_file() and p("ck/mad_sft_best.pt").is_file()
    state = torch.load(p("ck/mad_sft_latest.pt"), weights_only=True)
    assert state["metadata"]["stage"] == "instruction_tuning" and state["step"] == 8
    # interrupted fine-tune can resume
    run("finetune", "--sft", p("proc/sft.jsonl"), *common, "--steps", 10, "--resume", p("ck/mad_sft_latest.pt"))
    capsys.readouterr()
    run("ask", "Say", "hello", "--tokenizer", p("tok.json"),
        "--checkpoint", p("ck/mad_sft_latest.pt"), "--seed", 1, "--max-new-tokens", 8, "--threads", 1)
    run("status", "--config", p("c.json"), "--corpus", p("proc/corpus.txt"), "--tokenizer", p("tok.json"))
    assert "natural-language stage" in capsys.readouterr().out


def test_ask_warns_when_not_tuned_and_rejects_huge_request(tmp_path, capsys):
    raw, instr = make_project(tmp_path)
    p = lambda name: tmp_path / name
    run = lambda *a: cli_main([str(x) for x in a])
    run("init", "--config", p("c.json"), "--checkpoint-dir", p("ck0"))
    run("prepare-data", "--input", raw, "--output", p("proc/corpus.txt"))
    run("train-tokenizer", "--corpus", p("proc/corpus.txt"), "--sft", p("none.jsonl"),
        "--output", p("tok.json"), "--vocab-size", 400, "--config", p("c.json"))
    run("train", "--config", p("c.json"), "--tokenizer", p("tok.json"), "--corpus", p("proc/corpus.txt"),
        "--output-dir", p("ck"), "--steps", 2, "--seq-len", 32, "--batch-size", 2, "--grad-accum", 1,
        "--eval-every", 0, "--threads", 1)
    capsys.readouterr()
    run("ask", "hello there", "--checkpoint", p("ck/mad_latest.pt"), "--tokenizer", p("tok.json"),
        "--max-new-tokens", 3, "--threads", 1)
    assert "has not been instruction-tuned" in capsys.readouterr().out
    run("ask", "word " * 400, "--checkpoint", p("ck/mad_latest.pt"), "--tokenizer", p("tok.json"), "--threads", 1)
    assert "Shorten it" in capsys.readouterr().out
