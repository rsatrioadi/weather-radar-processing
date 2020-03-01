#include "udpbroadcast.h"

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 
#include <errno.h>

namespace udpbroadcast {

    udpclient::udpclient(int port): mPort(port) {
            
        // Creating socket file descriptor 
        if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
            throw "error sock";
        }
         
        int on=1;
        setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
        
        memset(&servaddr, 0, sizeof(servaddr)); 
          
        // Filling server information 
        servaddr.sin_family = AF_INET; 
        servaddr.sin_port = htons(mPort); 
        servaddr.sin_addr.s_addr = htonl(INADDR_BROADCAST);      
    }
    
    udpclient::~udpclient() {
        close(sockfd);
    }
    
    int udpclient::send(const char* message, size_t length) {
        return sendto(sockfd, message, length, MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr)); 
    }
    
    udpserver::udpserver(int port): mPort(port) {
            
        // Creating socket file descriptor 
        if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
            throw "error sock";
        } 
          
        memset(&servaddr, 0, sizeof(servaddr)); 
        memset(&cliaddr, 0, sizeof(cliaddr)); 
          
        // Filling server information 
        servaddr.sin_family    = AF_INET; // IPv4 
        servaddr.sin_addr.s_addr = htons(INADDR_ANY); 
        servaddr.sin_port = htons(mPort); 
          
        // Bind the socket with the server address 
        if ( bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0 ) { 
            throw "error bind";
        } 
    }
    
    udpserver::~udpserver() {
        close(sockfd);
    }
    
    int udpserver::recv(char* buffer, size_t length) {
        unsigned int len, n; 
        len = sizeof(cliaddr);  //len is value/resuslt
        n = recvfrom(sockfd, buffer, length, MSG_WAITALL, (struct sockaddr *) &cliaddr, &len); 
        return n;
    }
}
