#!/bin/bash
# test local machine using cpu


model="hemlgop homlgop hemlrn homlrn pop popfast popmemo popmemh"
version="py2 py3"


function test_local {
# function to perform testing on local machine
# usage test_local [cpu/gpu]
# environment py2_test or py3_test must exist!
    
    if [ $1 == 'cpu' ]; then
        computation=0
    else
        computation=3
    fi

    local test_dir="test_local_"$1"_outputs"

    if [[ -d ${test_dir} ]]; then
        rm -r ${test_dir}
    fi
    
    mkdir ${test_dir}
    
    for v in $version; do
        if source activate ${v}_$1_test ; then
            for m in $model; do
                python test_models.py -s $2 -m $m -i $computation -v $v > ${test_dir}/${m}_${v}_out.txt
            done
        else
	        echo "python environment ${v}_${1}_test does not exist!"
        fi	    
    done
}


function test_cluster {
# function to perform testing on cluster taito cpu, gpu or narvi cpu
# usage test_cluster [cpu/gpu]

    if [ $1 == 'cpu' ]; then
        local computation='cpu'
	local array="0,1"
    else
        local computation='gpu'
	local array="2,3"
        if [ hostname == 'narvi*' ]; then
            echo "Not support narvi gpu cluster yet!"
            return 0
	fi
    fi
    
    local test_dir="test_cluster_"${computation}"_outputs"
    if [[ -d ${test_dir} ]]; then
        rm -r ${test_dir}
    fi
    
    mkdir ${test_dir}
    
    for v in $version; do
        for m in $model; do
            local name=${m}_${v}
            local arr="%a"
            local out_file=${test_dir}/${m}_${v}_${arr}.o
            local err_file=${test_dir}/${m}_${v}_${arr}.e
            if [ hostname == 'narvi*' ]; then
                partition=normal
                configuration='source activate ${v}_test'
                python_cmd="python"
                constraint="#SBATCH --constraint=hsw\n"
                cluster_name="narvi"
            else
                if [ ${computation} == 'cpu' ]; then
                    partition=parallel
                    if [ ${v} == py2 ]; then
                        configuration='module purge \nmodule load python-env/2.7.10'
                    else
                        configuration='module purge \nmodule load python-env/3.5.3'
                    fi
                    constraint="#SBATCH --constraint=hsw\n"
                else
                    partition=gpu
                    if [ ${v} == py2 ]; then
                        configuration='#SBATCH --gres=gpu:p100:1 \nmodule purge \nmodule load python-env/2.7.10-ml'
                    else
                        configuration='#SBATCH --gres=gpu:p100:1 \nmodule purge \nmodule load python-env/3.5.3-ml'
                    fi
                    constraint=''
                    
                fi
                
                cluster_name="taito"
                python_cmd="srun python"
                
            fi
            
            bash_file=${test_dir}/${m}_${v}.sh
            if [[ -f ${bash_file} ]]; then
                rm ${bash_file}
            fi
            touch ${bash_file}
            
            local a='$'
            echo "#!/bin/bash" >> ${bash_file}
            echo "#SBATCH -J ${name}" >> ${bash_file}
            echo "#SBATCH -o ${out_file}" >> ${bash_file}  
            echo "#SBATCH -e ${err_file}" >> ${bash_file}
            echo "#SBATCH -t 2-00:00:00" >> ${bash_file}
            echo "#SBATCH --mem=16G" >> ${bash_file}
            echo "#SBATCH -c 4" >> ${bash_file}
            echo "#SBATCH -p ${partition}" >> ${bash_file}
            echo "#SBATCH --array="${array} >> ${bash_file}
            printf "${constraint}" >> ${bash_file}
            printf "${configuration}\n" >> ${bash_file}
            printf "${python_cmd} test_models.py -s True -m ${m} -i ${a}SLURM_ARRAY_TASK_ID -v ${v} -n ${cluster_name}\n" >> ${bash_file}
            
   	      sbatch ${bash_file} 
        done
    done
}

if [ $# -eq 0 ]; then
    echo "usage test_models.sh [local/cluster] [cpu/gpu] [source]"
    return 0
fi
if [ $# -eq 1 ]; then
    echo "usage test_models.sh [local/cluster] [cpu/gpu] [source]"
    return 0
fi

# test on local machine
if [ $1 == 'local' ]; then
    # create environment
    echo 'y' | conda create --name py2_${2}_test python==2.7.14
    echo 'y' | conda create --name py3_${2}_test python==3.5.3 

    # install dependencies for python 2 test environment
    if source activate py2_${2}_test; echo 'y' | pip install joblib dill ; then
        :
    else
        echo "Failed to activate py2_${2}_test or to install joblib, dill"
    fi

        
    if [ ${2} == 'cpu' ]; then
        if pip install tensorflow keras; then 
            :
        else
	        echo "Failed to install tensorflow and keras"
            return 1
	    fi
    else
        if pip install tensorflow-gpu keras; then
            :
        else
             echo "Failed to install tensorflow-gpu and keras"
             return 1
	    fi
    fi

    # install dependencies for python 3 test environment
    if source activate py3_${2}_test; echo 'y' | pip install joblib dill ; then
        :
    else
        echo "Failed to activate py3_${2}_test or to install joblib, dill"
    fi

        
    if [ ${2} == 'cpu' ]; then
        if pip install tensorflow keras; then 
            :
        else
	        echo "Failed to install tensorflow and keras"
            return 1
	    fi
    else
        if pip install tensorflow-gpu keras; then
            :
        else
             echo "Failed to install tensorflow-gpu and keras"
             return 1
	    fi
    fi

    source activate base
 
    if [ $# -eq 3 ]; then
        test_local $2 'True'
    else
        # build wheel
        if (
            cd ..
            cd ..
            python setup.py sdist bdist_wheel --universal            
            cp ./dist/*.whl ./test/models/
            rm -r ./*.egg-info dist build 
            ) ; then
            echo "Successfully build wheel"
        fi
    
       

        # install from wheel
        if source activate py2_${2}_test; pip install $(ls *.whl)  &&  source activate py3_${2}_test; pip install $(ls *.whl) ; then
            echo "Sucessfully install library from wheel" 
        else
            echo "Failed to install library from wheel"
            return 1
        fi
           
        # run the test from intalled library 
        if test_local $2 'False' ; then
            echo "Finish testing on local machine"
        else
            echo "Failed to test on local machine"
        fi 

        # uninstall from system package 
        if source activate py2_${2}_test; echo 'y' | pip uninstall pygop  &&  source activate py3_${2}_test; echo 'y' | pip uninstall pygop ; then
            echo
        else
            echo "Failed to uninstall library using pip"
            return 1
        fi

        # remove the generated wheel
        if rm $(ls *.whl); then
            :
        else
            echo "Failed to remove the generated wheel"
        fi 
    fi        
else
# test on cluster, only test from source
    test_cluster $2
fi



