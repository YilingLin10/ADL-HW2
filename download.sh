if [ ! -d "/cs" ]; then
  wget https://www.dropbox.com/s/owp7gz0g9zkrf30/cs.zip?dl=1 -O cs.zip
  unzip cs.zip
fi
if [ ! -d "/qa" ]; then
  wget https://www.dropbox.com/s/hvzlox96u53gimt/qa.zip?dl=1 -O qa.zip
  unzip qa.zip
fi