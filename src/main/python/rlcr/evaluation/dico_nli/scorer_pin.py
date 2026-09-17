"""Audited upstream revision and byte digests; upstream source is not vendored."""
REVISION = "588968e610197ddc4c440314701cbc587afc4c1b"
REPOSITORY = "https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI"
DOWNLOAD_ROOT = (
    "https://raw.githubusercontent.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/" + REVISION
)
DEFAULT_SCORER_DIR = ".cache/rlcr/dico-nli/" + REVISION
FILE_HASHES = {
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "evaluation_functions/__init__.py": "5e5ff6d6fd58b01638a8fd7ee7f482431bff837d134a0e79e5bfe0a5705baa15",
    "evaluation_functions/__main__.py": "f27cadcf05f172d2b853cb371ab3f99b098287f0ddbe0e19179df8a1143962fa",
    "evaluation_functions/aggregation.py": "a8ab594ca319c5c1eeeaa7307e427d028614cb65e37d5e95f2efaf60dfac4fc0",
    "evaluation_functions/cli.py": "00062311915e28170b71b4a524cf270381c6b8256660c45f3d182d603cb95540",
    "evaluation_functions/errors.py": "00173f95a00c80e8190a8e73f2b452856c11a8eaaee72608a03cffb47f2cb128",
    "evaluation_functions/io.py": "a87e368e154c4cd7350142d1af5a4c83041a81f68d18dbbc9b5f5bc2297c4cd1",
    "evaluation_functions/labels.py": "d500de7a8ef68460322516e4cbf431e19387e7555e9a4c2b2673dd727b17a98f",
    "evaluation_functions/metrics.py": "448c0fd72b149b091216b9a65bfbf88b3744aa124f20e5c241c51ff33f7efd7e",
    "evaluation_functions/schemas.py": "3a8cbca1ae93ba547d23bcf9c8eec0de2cf803f5fdc63166e7ad3fbb243c3193",
    "evaluation_functions/scorer.py": "076c5956e82fffe6234f1cecb3351f555c92f3d61798eab30f68e0116a7c038f",
    "evaluation_functions/templates.py": "ded61fc53ddbdebd25f27ed01730012693f742dd210654d61392603ce4892c27",
    "evaluation_functions/validation.py": "bb6258b41b1312fc88e5851bf9bb8615694c3600b0ef73643bc1cdff7720b9ea",
    "evaluation_functions/README.md": "39901ba138b27d5d0ae3af739e3314915a6db51d1b9ce04dc06dd9e49ad7b23d",
}
