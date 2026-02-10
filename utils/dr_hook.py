#-*- codind : utf-8 -*- 
import sys , os 



__all__=[ "DrHook"  ]

class DrHook:
    def __init__(self, on=None ):
        if on != None and on == True:
          self.on = True 

        return None 


    def CallDrHook(self, var, value ):
        self.varname=var
        self.value  =value  

        os.environ[self.varname]= value 
        return None 

    def SetDrHook(self): 
        dh_var={
        "DR_HOOK"                :"", 
        "DR_HOOK_OPT"            :"",
        "DR_HOOK_IGNORE_SIGNALS" :"",
        "DR_HOOK_PROFILE_LIMIT"  :"",
        "DR_HOOK_IGNORE_SIGNALS" :"",
        "DR_HOOK_CATCH_SIGNALS"  :"",
        "DR_HOOK_PROFILE_PROC"   :"",
        "DR_HOOK_HPMSTOP"        :"",
        "DR_HOOK_PROFILE"        :"",
        "DR_HOOK_HASHBITS"       :"",
         }
    

        if self.on:
           self.CallDrHook ( dh_var[var] , value )
         
        return None 
    def SetSignal(self):
        """
        Method : CAUGHTED SIGNALS 

        Selects the diffent signals that we can encouter when some of the 
        DR_HOOK_*  VARS are enabled ( especially DR_HOOK_OPT )
        The same SIG* signals could be used for the C part of this module 
        Depenfing on the UNIX distribution this list can be found under 
        /usr/include/bits/signals-generic.h or 
        /usr/include/x86_64-linux-gnu/bits/signum-generic.h
        """
        dh_sigs={ 
                  # GENERIC SIGNALS (COMPATIBLE WITH RedHat , OpenSuse and debian  )
                  # ISO C99 signals.
                  2  : "SIGINT"  ,              #/* Interactive attention signal.
                  4  : "SIGILL"  ,              #/* Illegal instruction.  
                  6  : "SIGABORT",              #/* Abnormal termination.  
                  8  : "SIGFPE"  ,              #/* Erroneous arithmetic operation.  
                 11  : "SIGSEGV" ,              #/* Invalid access to storage.  
                 15  : "SIGTERM" ,              #/* Termination request.  
                  # Historical signals specified by POSIX. 
                  1  : "SIGHUP"  ,              #/* Hangup.  
                  3  : "SIGQUIT" ,              #/* Quit.  
                  5  : "SIGTRAP" ,              #/* Trace/breakpoint trap.  
                  9  : "SIGKILL" ,              #/* Killed.  
                 10  : "SIGBUS"  ,              #/* Bus error.  
                 12  : "SIGSYS"  ,              #/* Bad system call.  
                 13  : "SIGPIPE" ,              #/* Broken pipe.  
                 14  : "SIGALRM" ,              #/* Alarm clock.  
                  # New(er) POSIX signals (1003.1-2008, 1003.1-2013).  
                 16  : "SIGURG"  ,              #/* Urgent data is available at a socket.  
                 17  : "SIGSTOP" ,              #/* Stop, unblockable.  
                 18  : "SIGTSTP" ,              #/* Keyboard stop.  
                 19  : "SIGCONT" ,              #/* Continue.  
                 20  : "SIGCHLD" ,              #/* Child terminated or stopped.  
                 21  : "SIGTTIN" ,              #/* Background read from control terminal.  
                 22  : "SIGTTOU" ,              #/* Background write to control terminal.  
                 23  : "SIGPOLL" ,              #/* Pollable event occurred (System V).  
                 24  : "SIGXCPU" ,              #/* CPU time limit exceeded.  
                 25  : "SIGXFSZ" ,              #/* File size limit exceeded.  
                 26  : "SIGVTALRM",             #/* Virtual timer expired.  
                 27  : "SIGPROF" ,              #/* Profiling timer expired.  
                 30  : "SIGUSR1" ,              #/* User- d signal 1.
                 31  : "SIGUSR2"                #/* User- d signal 2.   
                 }
