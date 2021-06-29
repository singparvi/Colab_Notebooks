FROM ubuntu

#declare username and userpass variable
ARG USERNAME=dockeruser
ARG USERPASS=1
ARG DEBIAN_FRONTEND=noninteractive

# update and upgrade environment
RUN apt-get -y  update && apt-get -y upgrade

# install packages
RUN apt -y install python3 python3-pip curl git sudo wget whois openssh-server

#RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
#    -t https://github.com/denysdovhan/spaceship-prompt \
#    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
#    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
#    -p git \
#    -p ssh-agent \
#    -p https://github.com/zsh-users/zsh-autosuggestions \
#    -p https://github.com/zsh-users/zsh-completions

# install heroku dependencies
#RUN pip3 install dash==1.9.1 dash-bootstrap-components gunicorn plotly joblib==0.17.0 scikit-learn==0.23.2 category-encoders==2.2.2 xgboost==1.3.3 dash-daq pandas==1.1.5

# Add a non-root user & set password
RUN useradd -ms /bin/bash $USERNAME
RUN sudo adduser $USERNAME sudo

# Set password for non-root user
RUN usermod --password $(echo "$USERPASS" | mkpasswd -s) $USERNAME

# Copy the entrypoint
COPY entrypoint.sh entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create the ssh directory and authorized_keys file
USER $USERNAME
RUN mkdir /home/$USERNAME/.ssh && touch /home/$USERNAME/.ssh/authorized_keys
USER root

# Set volumes
VOLUME /home/$USERNAME/.ssh
VOLUME /etc/ssh

# RUN apt -y install openssh-server

# Run entrypoint
CMD ["/entrypoint.sh"]

RUN mkdir /tmp/GitHubRepo
RUN cd /tmp/GitHubRepo
RUN git clone https://github.com/singparvi/Global-Precipitation.git