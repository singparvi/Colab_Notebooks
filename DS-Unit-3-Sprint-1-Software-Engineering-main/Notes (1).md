
# Module 1
Docker Initalization

docker build -t "local/python-ssh" --build-arg USERNAME=pythonssh --build-arg USERPASS=sshpass Dockerized-Python-SSH

docker run -d -p 2222:22 --name pythonssh local/python-ssh

ssh pythonssh@127.0.0.1 -p 2222


#make virtual environment and uploading the file on the pypi

documentation --> https://packaging.python.org/tutorials/packaging-projects/

video to watch --> https://www.youtube.com/watch?v=uzptI2Ny1Fc&list=PLUJxxNlebNAvFoPaGleFeew92D45xGaNy&index=15

python3 -m venv venv

source venv/bin/activate

virtualenv

pipenv shell	

# create conda environment

conda create --name py37 python==3.7



Pypi posting 
https://packaging.python.org/tutorials/packaging-projects/

pipenv install -d twine   # d means developer here

.#creating a wheel

python setup.py


### Terminal Commands

1. Make directory
mkdir -p /data/db
   
2. list 
ls
   
3. Remove directory
rm -rf  <name># r directory, f force
   
4. cd .. or cd \ for home

5. git -- version

6. which git  # finding which git is being used

7. pwd # gives working directory

8. cat <filename> # display file content in terminal

9. touch <file> # create a file


# Module 2

Keywords to use - linter, linting, pep8 - style, autopep8, REPL - Read-Eval-Print-Loop

video recording - https://www.youtube.com/watch?v=C4rCcb8wfSs

1. open . open the files in finder to location in current 

2. BareMinimumClass # upper camel case
    bareMinimumClass # camel case

3. Instantiating a class example:-
    b = BareMinimumClass()

4. move command in terminal
    mv <from> <to>
   
5. class BareMinimumClass:  $making a class
    class BareMinimumClass(): $ inheriting in the class
   
6. git and push
    ssh-keygen -f lambdagit # make a new key lambdagit
   


7. docker
    cd /tmp/pycharm_project_690/
   
   cat lambdagit.pub

    ssh -T git@github.com # test ssh connection with git

    ssh-keygen -t rsa -C "your_email@example.com" # generate ssh key as recommended by github rsa generated must be named rsa_id for git to know you correctly else will give you error.

    git config credential.helper store # store git credentials for next push

    git push --set-upstream origin master

    git push git@github.com:singparvi/lambda3.1.2demo.git # sample ssh push

    git push remote origin <url>

    nice AWS documentation --> https://docs.aws.amazon.com/codecommit/latest/userguide/how-to-mirror-repo-pushes.html  

    git checkout -b new_branch # make new branch

    git branch # look for branch in place

    git remote -v # display the remote linked with

    git commit -am "make it better" # commit with comment

    git push

    git clone --branch <branchname> <remote-repo-url>

    git push --set-upstream origin    # clone current branch to remote

    git pull --rebase  # dont know what it does

    git checkout -b new_branch # make new branch with name new_branch

    git push --set-upstream origin new_branch # open new branch on remote 
    
    git checkout main  # switch to main branch
   
    git reflog     # i think it goes back to the base repo. Clear all changes.
   
    git reset --hard 16e5b75 # go back to commit id

    export GIT_TRACE_PACKET=1
    export GIT_TRACE=1
    export GIT_CURL_VERBOSE=1   # All three lines above see versbose why things are failing
   
    brew install git-sizer # Filter bigger files

    To remove folder/directory only from git repository and not from the local try 3 simple commands.
    Steps to remove directory
    
    git rm -r --cached FolderName
    git commit -m "Removed folder from repository"
    git push origin master
    



8. Notes, keywords:
    namespace (__main__), attributes, methods, python classes are usually public unless mentioned private, mutable, immutable, 
   DRY (Don't Repeat Yourself), not WET (Write Every Time)
   
9. Check code style with PEP* Style Guide, using PEP8, black or flake8

10. docker copy files from container to host
    docker cp 4173eb4c08ee:/tmp/pycharm_project_690/Unit-3-Sprint-1-Software-Engineering-main/module2-oop-code-style-and-reviews/Assignment_2/helper_functions.py helper.py
    
11. docker cp 4173eb4c08ee:/tmp/pycharm_project_690/Unit-3-Sprint-1-Software-Engineering-main/module2-oop-code-style-and-reviews/Assignment_2/. /Users/rob/G_Drive_sing.parvi/Colab_Notebooks/Unit-3-Sprint-1-Software-Engineering-main/module2-oop-code-style-and-reviews/Assignment_2 # note the /. in SRC. Copies the complete directory from SRC to location


John's video --> https://youtu.be/CHLHqLP8odA


# MODULE 3

Docker Container links- 
https://training.play-with-docker.com/ops-s1-images/
https://github.com/ageron/handson-ml/tree/master/docker
https://docs.docker.com/engine/reference/commandline/docker/

Nick video -- https://www.youtube.com/watch?v=bbvS--t0i-s&t=5s

 pip3 install -r requirements.txt 
 you can use docker commit to “save” the dependencies to your image.
 
1. Docker commands

   docker
   docker run hello-world
   docker run -it ubuntu bash # get into interactively and gets in ubuntu bash
   docker ps -a # all docker IDs 
   docker ps # all **running** docker IDs
   ls /
   docker image ls
   docker build -t "local/python-ssh" --build-arg USERNAME=pythonssh --build-arg USERPASS=sshpass Dockerized-Python-SSH # build from docker image
   docker build -t "local/dockerassignment" --build-arg USERNAME=dockeruser --build-arg USERPASS=1 DockerAssignment # example
   docker run -d -p 2222:22 --name pythonssh local/python-ssh # run a container after build
   docker run -d -p 2222:22 --name dockerassignment local/dockerassignment # example
   docker kill <container>
   docker rm <container> # remove it 
   ssh pythonssh@127.0.0.1 -p 2222 # ssh into container
   ssh dockeruser@127.0.0.1 -p 2222 # example
   rm known host from .ssh if the hash change
   docker exec -it some-postgres bash
   

# MODULE 4

1. Pytest 
   pytest reddit_modul # pytest is better than unittest and is used more 
   python setup.py sdist bdist_wheel
   

2. https://www.sphinx-doc.org/en/master/ # document writer


   
    
