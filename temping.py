def get_temps(args, tokenizer):
    temps = {}
    with open(args.data_dir + '/' + 'temp.txt', 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            info = {}
            info['name'] = items[0].strip()
            info['temp'] = items[1:-1]  + ['[MASK]']
            info['label'] = items[-1]
            temps[info['name']] = info                                                
            #print(info)
    print(temps)
    return temps
