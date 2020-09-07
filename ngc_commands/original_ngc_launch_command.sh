cd original-transformer-xl-parallel \
  && aws configure \
  && pip install -r requirements.txt \
  && python launch.py --config=one_machine
