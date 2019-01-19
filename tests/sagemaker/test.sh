cd -- "$(dirname -- "${0}")"

# See http://unix.stackexchange.com/questions/101080/realpath-command-not-found
realpath ()
{
    f=$@;
    if [ -d "$f" ]; then
        base="";
        dir="$f";
    else
        base="/$(basename "$f")";
        dir=$(dirname "$f");
    fi;
    dir=$(cd "$dir" && /bin/pwd);
    echo "$dir$base"
}

docker run -v "$(realpath $1)"opt/ml:/opt/ml -it --rm ikeyasu/roboschool_chainerrl:latest train
