# Skill Boundary Agent

Skill Boundary Agent (SBA), a brand new frame to detect students' skill boundary automaticallly.

## Reproduce

make output directory:

```bash
mkdir outputs
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

then, all results will appear in `outputs` folder and console.

## Frame

- `cdm/`: pretrained model
- `data/`: test data and context info
- `utils/`: some tool functions, such as graph and llm_api function
- `config.py`: global config
- `sba_baseline.py`: algorithm baseline
- `sba_ours.py`: our SOTA algorithm
- `validate_downstream1.py`: validate downstream task of _Frontier-Question Recommendation_
- `validate_downstream2.py`: validate downstream task of _Target Skill Path Planning_
- `validate_hypothesis.py`: validate initial hypothesis
