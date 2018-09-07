# initialize git repo and make first commit
git init
git add .
git commit -a -m 'first commit'

# Read in config file variables using this function
function parse_yaml {

   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
eval $(parse_yaml config.yaml)

# Create conda environment with project name
echo y | conda create --name $project_name python jupyter
# Activate environment
source activate $project_name
# Attach kernel of this environment for use with jupyter 
python -m ipykernel install --user --name $project_name --display-name $project_name

# Install requirements 
make requirements

# Install utils from github
cd src
git clone https://github.com/safurrier/data-science-utils
mv data-science-utils utils
cd ..

# Checkout new branch for dev
#git checkout -b dailylab