# Data analysis examples

You can use the phy API to facilitate the analysis of ephys data in the file formats supported by phy. Read the full API documentation to see the list of properties and methods of the `TemplateModel` class.


## Plotting a template waveform


```python
import matplotlib.pyplot as plt
from phylib.io.model import load_model

# Load the TemplateModel instance.
model = load_model('params.py')

# Load a given template.
bunch = model.get_template(model.template_ids[0])

# For each channel, plot the template waveform on that channel.
for i, channel_id in enumerate(bunch.channel_ids):
    plt.subplot(1, len(bunch.channel_ids), i + 1)
    plt.plot(bunch.template[:, i])
    plt.title("Channel #%d" % channel_id)
plt.show()
```


## Plotting the template waveforms of a cluster

```python
import matplotlib.pyplot as plt
from phylib.io.model import load_model

model = load_model('params.py')
cluster_id = model.cluster_ids[0]  # first cluster

# This function takes into account the templates from which the cluster stems from,
# and computes a weighted average of the template waveforms depending on the number of spikes
# from each template.
bunch = model.get_cluster_mean_waveforms(cluster_id)

# Plot the cluster template on each "best" channel.
for i, channel_id in enumerate(bunch.channel_ids):
    plt.subplot(1, len(bunch.channel_ids), i + 1)
    plt.plot(bunch.mean_waveforms[:, i])
    plt.title("Channel #%d" % channel_id)
plt.show()
```
