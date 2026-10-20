## 学术资源加速
服务器位于中国大陆境内因而访问外部资源可能需要网络代理：

```plain
source /etc/network_turbo
```

访问国内资源时请重置代理避免卡顿和断连：

```plain
unset http_proxy && unset https_proxy
```

## 关机
实例按量计费，每一秒额外时间都会产生额外花费。因此，没有任务进行时应该将其关闭：

```plain
shutdown
```

执行以上命令进行关机。或者，使用全称：

```plain
/usr/bin/shutdown
```

在py脚本中，形式为：

```plain
os.system("/usr/bin/shutdown")
```

## 存储
保证两点：1.所有相关的实验结果数据文件必须在项目文件夹下可以访问，存储在项目文件夹下或通过symlink进行链接保证可访问性。2.大文件需要存储在数据盘autodl-tmp，防止爆系统盘root/。

