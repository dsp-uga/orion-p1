./setup.py bdist_egg

gcloud dataproc jobs submit pyspark \
    --cluster  cluster-2706\
    --region us-east1 \
    --py-files ./dist/orion-0.1.0.dev0-py3.6.egg \
    ./scripts/driver.py \
    -- $@
