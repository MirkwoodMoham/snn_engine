======
R-STDP
======




R-STDP constants

See [TODO:source].

..
    \alpha &=& 1 \\
    \beta &=& 0 \\

.. math::

    \begin{eqnarray}
        \Phi_r &=& 1 \\
        \Phi_p &=& 1 \\
        a_{rp} &=& 0.95 \\
        a_{pm} &=& -0.95 \\
        a_{rm} &=& -0.95 \\
        a_{pp} &=& 0.95 \\
    \end{eqnarray}




.. pcode::
    :linenos:
    :id: qq

    \begin{algorithm}
    \caption{Localized R-STDP}
    \begin{algorithmic}

    \PROCEDURE{Downstream-STDP}{$N_{src}, d, t$}

        \STATE $G_{src} = \text{Group}_N[N_{src}]$
        \STATE $\text{syn}_{\text{start}} = \text{SynDelayIndices}_N(N_{src}, d)$
        \STATE $\text{syn}_{\text{end}} = \text{SynDelayIndices}_N(N_{src}, d + 1) - 1$

        \FOR{$s$ = $\text{syn}_{\text{start}}$ to $\text{syn}_{\text{end}}$ }

            \STATE $N_{snk} = \text{Rep}_{\text{}N, S}[N_{src}, s]$
            \STATE $G_{snk} = \text{Group}_N[N_{snk}]$

            \IF{$(t - \text{LastFiringTime}_N[N_{snk}]) < d$
                \AND $\text{STDPConfig}_G[G_{src}, G_{snk}] \neq 0$}
                \STATE $\alpha = \text{STDPConfig}_G[G_{src}, G_{snk}] > 0$
                \STATE $\beta = !\alpha$
                \STATE $w = |\text{W}[N_{src}, N_{snk}]|$

                \IF{$w < 1$}

                    \IF{$\text{Type}_N[N_{snk}] == 2$}
                        \STATE $\text{W}_{N,S}[N_{src}, N_{snk}]$
                        += ($\alpha * \Phi_r * a_{rm} + \beta * \Phi_p * a_{pp}$) * w * (1 - w)
                    \ELSIF{$\text{Type}_N[N_{snk}] == 1$}
                        \STATE $\text{W}_{N,S}[N_{src}, N_{snk}]$
                        += ($\alpha * \Phi_r * a_{rp} + \beta * \Phi_p * a_{pm}$) * w * (1 - w)
                    \ENDIF

                \ENDIF
            \ENDIF

        \ENDFOR
    \ENDPROCEDURE
    \PROCEDURE{Uptream-STDP}{$N_{src}, d, t$}

    \IF{$d = 0$ \AND \NOT $\text{IsSensory}(G_{src})$}
        \STATE $\text{syn}_{\text{start}} =  \text{Idcs}_{pre, N}[N_{src}]$
        \STATE $\text{syn}_{\text{end}} =  \text{Idcs}_{pre, N}[N_{src} + 1] - 1$

        \FOR{$s$ = $\text{syn}_{\text{start}}$ to $\text{syn}_{\text{end}}$}
            \STATE $i = \text{Rep}_{\text{}pre}[s]$
            \STATE $w = |\text{flatten}(\text{W}_{N,S})[i]| ( = |\text{W}_{N,S}[N_{pre\text{-}src}, N_{src}]|)$
            \IF{$w < 1$}
                \STATE $N_{pre\text{-}src} = \text{int}(i/S)$
                \STATE $G_{pre\text{-}src} = \text{Group}_N[N_{pre\text{-}src}]$

                \IF{$(t - \text{LastFiringTime}_N[N_{pre\text{-}src}]) < 2 * d$
                    \AND $\text{STDPConfig}_G[G_{pre\text{-}src}, G_{src}] \neq 0$}

                    \STATE $\alpha = \text{STDPConfig}_G[G_{pre\text{-}src}, G_{src}] > 0$
                    \STATE $\beta = !\alpha$

                    \IF{$\text{Type}_N(N_{src}) == 2$}
                        \STATE $\text{W}_{N,S}[N_{pre\text{-}src}, N_{src}]$
                        += ($\alpha * \Phi_r * a_{rp} + \beta * \Phi_p * a_{pm}$) * w * (1 - w)

                    \ELSIF{$\text{Type}_N[N_{src}] == 1$}
                        \STATE $\text{W}_{N,S}[N_{pre\text{-}src}, N_{src}]$
                        += ($\alpha * \Phi_r * a_{rm} + \beta * \Phi_p * a_{pp}$) * w * (1 - w)

                    \ENDIF
                \ENDIF
            \ENDIF
        \ENDFOR

    \ENDIF
    \ENDPROCEDURE
    \end{algorithmic}
    \end{algorithm}



