# Linux   
## High Performance Computing (HPC) Cluster Login   

on Mac, open Terminal:

~~~
## ex) ssh janedoe@ssh.eri.ucsb.edu  
ssh {your_username}@ssh.{host}
{your_pwd}   
## ex) ssh janedoe@bellows.eri.ucsb.edu  
ssh {your_username}@{hpc_machine}.{host}
{your_pwd}   
~~~

Windows Applications:  
* Putty  
* MobaXterm (nice GUI)  


## [Install Linux on Windows](https://learn.microsoft.com/en-us/windows/wsl/install): 
- Launch PowerShell, enter ``` wsl --install```, download and double-click [update installer](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package) to run, restart computer  

### Download pip (general package installer) and venv (virtual environment manager) in Ubuntu 
- Launch Ubuntu in Start Menu:  
~~~
sudo apt update && sudo apt upgrade -y ## system updates 
sudo apt install python3-pip -y ## install pip 
sudo apt install python3-venv -y ## install venv 

~~~

[Create Virtual Machine (VM) on Windows](https://techcommunity.microsoft.com/t5/educator-developer-blog/step-by-step-how-to-create-a-windows-11-vm-on-hyper-v-via/ba-p/3754100)




## Commands   

### Tips:   

- pressing tab following a command (```cd``` , ```ls```, ```vim```) will list all possible files/folders and autofill if there's only one match 
- run ```pwd``` to print working directory and easily copy+paste path later

```|```   : pipe that takes first command's output and feeds it into the command after the pipe  
```*``` : wildcard matching any character  
```-r``` : recursively search subdirectories

### Examples: 
<b>list</b> ```-l``` long list, sorted by last modified time ```- t```, reversed ```- r``` --  such that last edited file is printed in the terminal last:  
> ls -ltr  

<b>list</b> two last modified files:  
> ls -ltr | tail -2

<b>size</b> of current directory:  
> du -sh .

<b>files sorted based on size</b>:  
> du -sh -- * | sort -rh  

<b>count</b> number of files in current directory:  
> ls | wc -l  

<b>count</b> number of files in directory (that start with 004):   
> ls -dq 004* | wc -l 

<b>move </b> files that contain 'filterstring' (in current dir) and move them into out_dir:  
> mv * filterstring * out_dir  

<b>zip</b> files in folder  
> zip -r out_file.zip in_dir*

<b>delete</b> files in current directory that start with S1_:  
> find . -type f -name 'S1_*' -delete  
> find . -name gee -exec ls {} \;
> find . -name gee -exec rm -rf {} \;  

<b>delete</b> files <i>recursively</i> in current directory that start with L3A_LC):   
> find . -name "L3A_LC*" -type f -exec rm -r {} +

<b>delete</b> folders in current directory that match string (contain the string 'landsat'):  
> find . -name "*landsat*" -type d -exec rm -r {} +

---




## SLURM Workload Manager
SLURM handles which bash scripts, tasks, or jobs submitted by all users connected to a HPC cluster get sent to which nodes, and when.

These bash scripts must be located in a user's ```~/code/bash``` directory, and they may contain lines to activate virtual environments and run python scripts.  

Submit a bash script to SLURM: 
~~~
cd ~/code/bash 
sbatch {bash_script_name}.sh
~~~

Check on bash script progress:
~~~
squeue ## to make sure it's running, see how busy the cluster is    
ls -ltr ## to    
cat {bottom_file.err} 
~~~
from ~/code/bash, prints the last file to be created printed at the bottom.   
copy filename, bottom_file.err   
cat command prints the contents of the file, bottom_file.err, in the terminal        



---

