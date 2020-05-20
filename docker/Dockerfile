
FROM python:3.7

# Define build variables.
ARG WORKSPACE=/workspace

# Create workspace and copy app data.
RUN mkdir $WORKSPACE
WORKDIR $WORKSPACE
# Copy necessary data only.
ADD recipipe $WORKSPACE/recipipe
ADD examples $WORKSPACE/examples
ADD tests $WORKSPACE/tests
COPY requirements* $WORKSPACE/
COPY setup.py $WORKSPACE/
COPY README.md $WORKSPACE/
COPY LICENSE $WORKSPACE/

# Install dev requirements.
RUN pip install -r requirements_dev.txt
# Install recipipe module.
RUN pip install .

# Install convenience scripts.
COPY docker/notebook /usr/bin
RUN chmod +x /usr/bin/notebook

