import json
import yaml
import copy

from helpers.k8s_yaml_obj import ConversionType
from helpers.k8s_yaml_obj import K8Deployment, K8Job, K8Selector, K8ResourceHandling, K8Container, K8Volume
from helpers.k8s_yaml_obj import K8Template, K8Service, K8Port

def create_worker(
    app_name,
    version,
    mounted_vol_name,
    share_name,
    registry,
    image_name,
    connection_port,
    tf_config):

    metadata = {"name": app_name}


    res_deploy = K8Deployment(metadata)

    identifiers = {"app": app_name, "env": "dev"}

    k8s_sel = K8Selector(identifiers)

    env_var_list = []
    env_var_list.append(["MY_VERSION", version, ConversionType.quoted_string])
    env_var_list.append(["MY_ENVIRONMENT", "dev", ConversionType.normal])
    env_var_list.append(["IS_DOCKER", "y", ConversionType.quoted_string])
    env_var_list.append(["TRAIN_MODEL", "y", ConversionType.quoted_string])
    env_var_list.append(["TRAINING_EPOCHS", 50, ConversionType.integer_string])
    env_var_list.append(["TRAINING_SAVE_FREQ", 3, ConversionType.integer_string])

    js_string = json.dumps(tf_config, separators=(',', ':'))
    env_var_list.append(["TF_CONFIG", js_string, ConversionType.normal])

    env_var_list_runtime = []
    env_var_list_runtime.append(["MY_NODE_NAME", "spec.nodeName"])
    env_var_list_runtime.append(["MY_POD_NAME", "metadata.name"])

    vol_mounts_list = []
    vol_mounts_list.append([mounted_vol_name, "/mnt/persistentstorage"])

    res_req = K8ResourceHandling("3000m", "4000Mi")

    container_list = [
        K8Container(
            app_name, 
            registry,
            image_name,
            version,
            env_var_list,
            env_var_list_runtime,
            vol_mounts_list,
            res_req)
        ]
    volume_list = [K8Volume(mounted_vol_name, "storage", share_name)]

    tmpl_spec = res_deploy.get_template_spec(container_list, volume_list)
    k8s_template = K8Template(identifiers, tmpl_spec)

    res_deploy.fill_specification(1, 1, k8s_sel, k8s_template)

    res_svc = K8Service(metadata)

    port_list = []
    port_list.append(K8Port(app_name, connection_port, connection_port))
    res_svc.fill_specification(port_list, identifiers)

    return [res_svc, res_deploy]

app_name = "imagesegmentation"
mounted_vol_name = "mountedvolume"

secret_storage_name = "storage"
share_name = "imagesegmentationdistributed"

registry = ""
image_name = "image_segmentation_distributed"
version = "1.0.3"

worker_list = []
for idx in range(3):
    worker_list.append(["worker{0}".format(idx), 5000 + idx])

tmp_tf_config = {}
tmp_tf_config["cluster"] = {}
tmp_tf_config["cluster"]["worker"] = ["{0}:{1}".format(www[0], www[1]) for www in worker_list]
tmp_tf_config["task"] = {'type': 'worker', 'index': 0}

for www in worker_list:
    local_tf_config = copy.deepcopy(tmp_tf_config)
    www.append(local_tf_config)
    tmp_tf_config["task"]["index"] += 1

all_resources = []
for www in worker_list:
    all_resources.extend(
        create_worker(
            www[0],
            version,
            mounted_vol_name,
            share_name,
            registry,
            image_name,
            www[1],
            www[2]))

file_path = "all_workers.yml"

tag_strings = []
tag_strings.append("!Service")
tag_strings.append("!Deployment")
with open(file_path, "w") as f:
    # yaml.dump(resource, f)
    yml_string = yaml.dump_all(all_resources, None)
    for sss in tag_strings:
        yml_string = yml_string.replace(sss, "")
    if yml_string.startswith("\n"):
        yml_string = yml_string[1:]
    f.write(yml_string)

# with open("my_first.yml", "r") as f:
#     allo = yaml.load(f)
#     print(allo.metadata)

