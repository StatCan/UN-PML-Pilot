#
"""
Simple Paillier Homomorphic Encrypted Aggregation.

Implement secure aggregation strategy based on flower implementation
https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py
"""
from __future__ import annotations
import sys
import json
import pickle
import datetime
from multiprocessing import Pool
from functools import reduce
import numpy as np
import phe
from typing import Callable, Dict, List, Optional, Tuple, Iterable, Union
from click import secho
from flwr.common import (
    # EvaluateIns,
    # EvaluateRes,
    # FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import fedavg

EncryptedNumberType = phe.EncryptedNumber
PublicKeyType = phe.PaillierPublicKey
PrivateKeyType = phe.PaillierPrivateKey
PlainIterable = Iterable[Union[int, float]]
EncryptedIterable = Iterable[EncryptedNumberType]


def encrypt_iterable(public_key, x: PlainIterable):
    """Encrypt an iterable of floats or integers.

    Parameters
    ----------
    public_key
        The public key object
    x
        Iterable of floats
    Returns
    -------
    map of EncryptedNumber
    """
    # return map(public_key.encrypt, x)
    with Pool(12) as p:
        return p.map(public_key.encrypt, x)


def decrypt_iterable(private_key, x: EncryptedIterable) -> PlainIterable:
    """Encrypt an iterable of floats or integers.

    Parameters
    ----------
    private_key
        The private key object
    x
        Iterable of encrypted objects
    Returns
    -------
    list of decrypted numbers
    """
    # return [private_key.decrypt(i) for i in x]
    with Pool(12) as p:
        return p.map(private_key.decrypt, x)


def serialize_encrypted(enc: EncryptedNumberType, exponent: int) -> str:
    """Serialize an encrypted number.

    Parameters
    ----------
    private_key
        The private key
    x
        Iterable of encrypted objects
    Returns
    -------
    list of decrypted numbers
    """
    if enc.exponent > -32:
        enc = enc.decrease_exponent_to(-32)
        assert enc.exponent == -32
    # else:
    #     print("Exponent is less than -32")
    # print("Exponent ", enc.exponent, type(enc.ciphertext()))
    return str(enc.ciphertext())


def load_encrypted_number(enc: int, exponent: int, public_key: PublicKeyType):
    """Load an encrypted number object.

    Parameters
    ----------
    enc
        encrypted serialized number as integer
    exponent
        The exponent (precision)
    public_key
        The public key
    Returns
    -------
    list of decrypted numbers
    """
    den = phe.EncryptedNumber(public_key, enc, exponent)
    return den


class EncArray(np.ndarray):
    """Define an array for encryption/decryption.

    This subclasses numpy ndarray, that allow us to leverage all the
    calculation machinery of numpy on cleartext and attempts to extend
    for addition and multiplication in case of encrypted numbers where
    applicable.
    """

    def __new__(cls, input_array):
        """Create a new EncArray from an iterable."""
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        """Add a custom attribute to this subclass EncArray."""
        if obj is None:
            return
        self.enc = getattr(obj, "enc", None)

    def __add__(self, other):
        """Override addition to allow encrypted objects."""
        if other.enc or self.enc:
            ret = EncArray([a + b for a, b in zip(self, other)])
            ret.enc = True
            return ret
        else:
            return super().__add__(np.asarray(other))

    def encrypt(self, public_key) -> EncArray:
        """Encrypt this array."""
        # using tolist to force convert numpy float types to native float
        # flatten and reshape
        enc = list(encrypt_iterable(public_key, self.flatten().tolist()))
        dens = EncArray(enc).reshape(self.shape)
        dens.enc = True
        return dens

    def decrypt(self, private_key) -> EncArray:
        """Decrypt this array."""
        if self.enc:
            array = EncArray(decrypt_iterable(private_key, self.flatten().tolist()))
            return array.reshape(self.shape)
        else:
            print("Not encrypted")

    def serialize(self, exp: int = -32) -> List:
        """Serialize EncArray of ciphertext objects.

        Returns
        -------
        List
            List with serialized ciphertext objects as strings
        """
        # if self.enc:
        return [serialize_encrypted(e, exp) for e in self.flatten()]

    def serialize_ndarray(self, exp: int = -32) -> np.ndarray:
        """Serialize EncArray of ciphertext objects.

        Returns
        -------
        numpy.ndarray
            ndarray with serialized ciphertext objects as strings
        """
        # if self.enc:
        return np.asarray(
            [serialize_encrypted(e, exp) for e in self.flatten()]
        ).reshape(self.shape)

    @classmethod
    def deserialize(
        cls, enc: Iterable[str], public_key: phe.PaillierPublicKey, exp=-32
    ) -> EncArray:
        """Deserialize an iterable of strings and return EncArray.

        Parameters
        ----------
        enc
            Iterable of strings, serialized ciphertexts
        public_key
            The public key used to get the ciphertexts

        Returns
        -------
        EncArray
            Array with ciphertext objects
        """
        try:
            earray = EncArray(
                [load_encrypted_number(int(e), exp, public_key) for e in enc]
            )
            earray.enc = True
            return earray
        except Exception as e:
            print(e)

    @classmethod
    def deserialize_ndarray(
        cls, enc: np.ndarray, public_key: phe.PaillierPublicKey, exp=-32
    ) -> EncArray:
        """Deserialize a numpy ndarray of strings and return EncArray.

        Parameters
        ----------
        enc
            ndarray with serialized ciphertext objects as strings
        public_key
            The public key used to get the ciphertexts

        Returns
        -------
        EncArray
            Array with ciphertext objects
        """
        try:
            earray = EncArray(
                np.asarray(
                    [
                        load_encrypted_number(int(e), exp, public_key)
                        for e in enc.flatten()
                    ]
                ).reshape(enc.shape)
            )
            earray.enc = True
            return earray
        except Exception as e:
            print(e)


class KeyGenerator(object):
    """Implement key generator for Paillier cryptosystem."""

    def __init__(self, key_length: int = 1024) -> None:
        """Create key pair.

        Parameters
        ----------
        key_length
            Length for the key.
        """
        self.key_length = key_length
        self.generate = True
        self.public_key, self.private_key = None, None
        self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.public_key, self.private_key = phe.generate_paillier_keypair(
            n_length=self.key_length
        )

    @classmethod
    def load_pkd(cls) -> KeyGenerator:
        """Load keys from pickle, do not generate them."""
        # do not call __init__
        obj = cls.__new__(cls)
        obj.load_keys_pkd()
        return obj

    @classmethod
    def load(cls) -> KeyGenerator:
        """Load keys from JWK (JSON) files. Do not generate new keys."""
        # do not call __init__
        obj = cls.__new__(cls)
        obj.load_keys()
        return obj

    @classmethod
    def load_public_pkd(cls, filename: str = "public.key") -> KeyGenerator:
        """Load public key only from pickle. Do not generate new keys."""
        # do not call __init__
        obj = cls.__new__(cls)
        obj.load_public_key_pkd(filename)
        return obj

    @classmethod
    def load_public(cls, filename: str = "public_key.json") -> KeyGenerator:
        """Load public key from JWK (JSON) file. Do not generate new keys."""
        # do not call __init__
        obj = cls.__new__(cls)
        obj.load_public_key(filename)
        return obj

    def kid_metadata(self, kind: str) -> str:
        """Generate kid metadata string."""
        if self.generate:
            return f"Paillier {kind} key generated "
            +f"by simple_pailler on {self.date}"
        else:
            return f"Paillier {kind} key loaded "
            +"by simple_pailler on {self.date}"

    def public_key_todict(self):
        """Export public key to python dict."""
        jwk_public = {
            "kty": "DAJ",
            "alg": "PAI-GN1",
            "key_ops": ["encrypt"],
            "n": phe.util.int_to_base64(self.public_key.n),
            "kid": self.kid_metadata("public"),
        }
        return jwk_public

    def public_key_fromdict(self, public_key):
        """Import public key from python dict."""
        error_msg = "Invalid public key"
        assert "alg" in public_key, error_msg
        assert public_key["alg"] == "PAI-GN1", error_msg
        assert public_key["kty"] == "DAJ", error_msg
        n = phe.util.base64_to_int(public_key["n"])
        self.public_key = phe.PaillierPublicKey(n)

    def private_key_todict(self):
        """Export private key to python dict."""
        jwk_public = self.public_key_todict()
        jwk_private = {
            "kty": "DAJ",
            "key_ops": ["decrypt"],
            "p": phe.util.int_to_base64(self.private_key.p),
            "q": phe.util.int_to_base64(self.private_key.q),
            "pub": jwk_public,
            "kid": self.kid_metadata("public"),
        }
        return jwk_private

    def save_public_key_pkd(self, filename: str = "public.key") -> None:
        """Save the public key using pickle.

        Parameters
        ----------
        filename
            The name for public key file
        """
        with open(filename, "wb") as fp:
            pickle.dump(self.public_key, fp)

    def save_public_key(self, filename: str = "public_key.json") -> None:
        """Save the public key in a JWK format, JSON file.

        Parameters
        ----------
        filename
            The name for public key json file
        """
        with open(filename, "w", encoding="utf8") as json_file:
            json.dump(self.public_key_todict(), json_file, allow_nan=False)

    def save_private_key_pkd(self, filename: str = "private.key") -> None:
        """Save the private key using pickle.

        Parameters
        ----------
        filename
            The name for private key
        """
        with open(filename, "wb") as fp:
            pickle.dump(self.private_key, fp)

    def save_private_key(self, filename: str = "private_key.json") -> None:
        """Save the private key in a JWK format, JSON file.

        Parameters
        ----------
        filename
            The name for private key json file
        """
        with open(filename, "w", encoding="utf8") as json_file:
            json.dump(self.private_key_todict(), json_file, allow_nan=False)

    def save_keys_pkd(self) -> None:
        """Save the public and private keys using pickle."""
        self.save_public_key_pkd()
        self.save_private_key_pkd()

    def save_keys(self) -> None:
        """Save the public and private keys in JWK format, JSON files."""
        self.save_public_key()
        self.save_private_key()

    def load_public_key_pkd(self, filename: str = "public.key"):
        """Load the public key using pickle.

        Parameters
        ----------
        filename
            The name for public key file
        """
        with open(filename, "rb") as fp:
            self.generate = False
            self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.public_key = pickle.load(fp)

    def load_public_key(self, filename: str = "public_key.json") -> None:
        """Load the public key from JSON file, JWK format.

        Parameters
        ----------
        filename
            The name for public key json file
        """
        with open(filename, "r", encoding="utf8") as json_file:
            self.generate = False
            self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            public_key = json.load(json_file)
            error_msg = "Invalid public key"
            assert "alg" in public_key, error_msg
            assert public_key["alg"] == "PAI-GN1", error_msg
            assert public_key["kty"] == "DAJ", error_msg
            n = phe.util.base64_to_int(public_key["n"])
            self.public_key = phe.PaillierPublicKey(n)

    def load_private_key_pkd(self, filename: str = "private.key"):
        """Load the private key using pickle.

        Parameters
        ----------
        filename
            The name for private key file

        Returns
        -------
        key
        """
        with open(filename, "rb") as fp:
            self.generate = False
            self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.private_key = pickle.load(fp)

    def load_private_key(self, filename: str = "private_key.json") -> None:
        """Load the private key from JSON file, JWK format.

        Parameters
        ----------
        filename
            The name for private key json file
        """
        with open(filename, "r", encoding="utf8") as json_file:
            self.generate = False
            self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            private_key = json.load(json_file)
            error_msg = "Public key not found in private key"
            assert "pub" in private_key, error_msg
            self.public_key_fromdict(private_key["pub"])
            private_key_error = "Invalid private key"
            assert "key_ops" in private_key, private_key_error
            assert "decrypt" in private_key["key_ops"], private_key_error
            assert "p" in private_key, private_key_error
            assert "q" in private_key, private_key_error
            assert private_key["kty"] == "DAJ", private_key_error
            _p = phe.util.base64_to_int(private_key["p"])
            _q = phe.util.base64_to_int(private_key["q"])
            self.private_key = phe.PaillierPrivateKey(self.public_key, _p, _q)

    def load_keys_pkd(self):
        """Load the public and private keys using pickle."""
        self.load_private_key_pkd()
        self.public_key = self.private_key.public_key

    def load_keys(self):
        """Load the public and private keys from JWK, JSON file."""
        self.load_private_key()


class SimplePaillierAvg(fedavg.FedAvg):
    """Implement secure aggregation strategy.

    Extending the FedAvg class...
    """

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """Implement Simple Paillier Averaging strategy.

        Implementation based on flower FedAvg

        Parameters
        ----------
        fraction_fit
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn
            Optional function used for validation. Defaults to None.
        on_fit_config_fn
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn
            Function used to configure validation. Defaults to None.
        accept_failures
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters
            Initial global model parameters.
        """
        super().__init__(
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters,
        )
        self.other = 1

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        secho("Calling aggregate fit", fg="green")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Load keys
        keygen = KeyGenerator().load_public()
        # Convert results
        res = []
        for client, fit_res in results:
            # bytes to ndarray
            res_array = parameters_to_weights(fit_res.parameters)
            weights_results = []
            # iterate in list of arrays from each client
            for n, e in enumerate(res_array):
                # check for encrypted serialized array
                if e.flatten().dtype.type is np.str_:
                    secho(f"Deserializing {e.size} elements ", fg="cyan",
                          nl=False)
                    enc_array = EncArray.deserialize_ndarray(e,
                                                             keygen.public_key)
                    secho(f"with shape {enc_array.shape}", fg="cyan")
                    weights_results.append(enc_array)
                else:
                    weights_results.append(e)
            res.append((weights_results, fit_res.num_examples))
        num_examples_total = sum([num_examples for _, num_examples in res])

        for n, e in enumerate(res):
            if isinstance(e, EncArray):
                print(f"Exponent {e[0].exponent}")
                break
        # Create a list of weights, each multiplied by the related
        # number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in res
        ]
        # Compute average weights of each layer
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        # serialization for transmission
        weights_pp = []
        for n, e in enumerate(weights_prime):
            if isinstance(e, EncArray):
                weights_pp.append(e.serialize_ndarray())
            else:
                weights_pp.append(e)
        return weights_to_parameters(weights_pp), {}

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        secho(
            "Calling evaluate... Not implemented. Server cant see the data!", fg="red"
        )
        return None
