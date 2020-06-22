#ifndef tcpclientserver_h
#define tcpclientserver_h

#include <netinet/in.h>

namespace tcp {
    
    class tcpclient {
        private:
            int mPort;
            int sockfd;
            struct sockaddr_in servaddr;
        public:
            tcpclient(int);
            ~tcpclient();
            
            int sendit(const char*,size_t);
    };   
    
    class tcpserver {
        private:
            int mPort;
            int sockfd; 
            struct sockaddr_in servaddr, cliaddr;
        public:
             tcpserver(int);
             ~tcpserver();
             
             int recv(char*,size_t);
    };
    
}

#endif // tcpclientserver_h

