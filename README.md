# Skill Boundary Agent

Skill Boundary Agent (SBA), a brand new frame to detect students' skill boundary automatically.

## Reproduce

Install Git and Git LFS, then clone the repository.

Install uv, then sync the python environment:

```bash
uv sync
```

set llm api:

```bash
echo "API_BASE=<your_llm_api_base_url>" > .env
echo "API_KEY=<your_llm_api_key>" >> .env
```

run and validate:

```bash
python sba_baseline.py
python sba_ours.py
```

then, check all results in `outputs` folder and console.

## Frame

- `cdm/`: pretrained model and model inference code
- `data/`: test data and context info
- `utils/`: tool functions, such as graph and llm_api function
- `config.py`: global config
- `sba_baseline.py`: baseline algorithms
- `sba_ours.py`: our SOTA algorithm
- `validate_downstream1.py`: validate downstream task of _Frontier-Question Recommendation_
- `validate_downstream2.py`: validate downstream task of _Target Skill Path Planning_
- `validate_hypothesis.py`: validate initial hypothesis
