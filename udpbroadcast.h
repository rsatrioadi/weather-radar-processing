#ifndef udpclientserver_h
#define udpclientserver_h

#include <netinet/in.h>

namespace udpbroadcast {
    
    class udpclient {
        private:
            int mPort;
            int sockfd;
            struct sockaddr_in servaddr;
        public:
            udpclient(int);
            ~udpclient();
            
            int send(const char*,size_t);
    };   
    
    class udpserver {
        private:
            int mPort;
            int sockfd; 
            struct sockaddr_in servaddr, cliaddr;
        public:
             udpserver(int);
             ~udpserver();
             
             int recv(char*,size_t);
    };
    
}

#endif // udpclientserver_h

